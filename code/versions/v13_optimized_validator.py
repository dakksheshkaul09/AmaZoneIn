import pandas as pd
import re
import unicodedata
import gc
import heapq
import time
import os
import psutil
import concurrent.futures
import numpy as np
import lightgbm as lgb

from collections import Counter, defaultdict
from rapidfuzz import fuzz

# =========================================================
# SETTINGS
# =========================================================
TARGET_CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 100

THRESHOLDS = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.75, 0.80, 0.82,
    0.84, 0.86, 0.88, 0.90, 0.92,
    0.94, 0.95, 0.96, 0.97, 0.98, 0.99
]
WORKERS = os.cpu_count() or 4

def get_ram_gb(): 
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# =========================================================
# NORMALIZATION & UTILS
# =========================================================
LEGAL_SUFFIX_MAP = {
    "pvt": "private", "private": "private", "ltd": "limited", "limited": "limited",
    "corp": "corporation", "corporation": "corporation", "inc": "incorporated", 
    "incorporated": "incorporated", "co": "company", "company": "company",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp"
}
LEGAL_SUFFIXES = set(LEGAL_SUFFIX_MAP.values())

ADDRESS_MAP = {
    "rd": "road", "road": "road", "st": "street", "street": "street", "ave": "avenue", 
    "av": "avenue", "avenue": "avenue", "dr": "drive", "drive": "drive", "ln": "lane", 
    "lane": "lane", "blvd": "boulevard", "boulevard": "boulevard", "hwy": "highway", 
    "highway": "highway", "pkwy": "parkway", "parkway": "parkway", "ct": "court", 
    "court": "court", "cir": "circle", "circle": "circle", "pl": "place", "place": "place"
}

def normalize_text(x):
    if pd.isna(x): return ""
    x = str(x).casefold()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = "".join(" " if unicodedata.category(c).startswith("P") else c for c in x)
    return " ".join(x.split())

def tokens(x): return set(re.findall(r"\w+", x)) if x else set()

def strong_name_core(x):
    x = normalize_text(x)
    if not x: return ""
    norm = " ".join(LEGAL_SUFFIX_MAP.get(t, t) for t in x.split())
    return " ".join(sorted(set(norm.split()) - LEGAL_SUFFIXES))

def strong_address(x):
    x = normalize_text(x)
    if not x: return ""
    return " ".join(sorted({ADDRESS_MAP.get(t, t) for t in x.split()}))

def char3grams(x):
    compact = "".join(c for c in x if c.isalnum()) if x else ""
    if len(compact) < 3: return set()
    return {compact[i:i + 3] for i in range(len(compact) - 2)}

def entity_f05(true_count, predicted_count, tp):
    if true_count == 0: return 1.0 if predicted_count == 0 else 0.0
    if predicted_count == 0: return 0.0
    precision = tp / predicted_count
    recall = tp / true_count
    if precision == 0 or recall == 0: return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

# =========================================================
# SINGLE-PASS VALIDATOR EXECUTION
# =========================================================
if __name__ == "__main__":
    total_start = time.time()
    
    if not os.path.exists("models/lgb_model.txt"):
        raise FileNotFoundError("models/lgb_model.txt not found.")

    print(f"[{get_ram_gb():.2f} GB] Loading Model & Validation Setup...")
    model = lgb.Booster(model_file="models/lgb_model.txt")

    val_ids = set(pd.read_csv("dataset/train/validation_s1_ids.tsv", sep="\t")["source1_entity_id"])
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1 = s1[s1["entity_id"].isin(val_ids)].copy()

    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt["source1_entity_id"].isin(val_ids)]

    true_count_by_s1 = {sid: 0 for sid in s1["entity_id"]}
    true_pairs_by_s1 = defaultdict(set)
    for row in gt.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids: continue
        tids = set(row.matched_entity_ids.split(","))
        true_count_by_s1[row.source1_entity_id] = len(tids)
        true_pairs_by_s1[row.source1_entity_id] = tids
    del gt; gc.collect()

    s1_map = {}
    s1_ids_list = []
    exact_n_lookup, core_n_lookup, str_a_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
    rare_n_lookup, rare_a_lookup = defaultdict(list), defaultdict(list)
    n_gram_lookup, a_gram_lookup = defaultdict(list), defaultdict(list)

    print(f"[{get_ram_gb():.2f} GB] PASS 1: Analyzing S1 Tokens & Building Index...")
    name_counts, addr_counts = defaultdict(Counter), defaultdict(Counter)
    name_gram_counts, addr_gram_counts = defaultdict(Counter), defaultdict(Counter)

    for i, row in enumerate(s1.itertuples(index=False)):
        c, sid = row.country, row.entity_id
        old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
        core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)

        s1_map[sid] = {
            "country": c,
            "old_name": old_n, "old_addr": old_a,
            "old_name_tokens": tokens(old_n), "old_address_tokens": tokens(old_a),
            "core_name": core_n, "strong_address": str_a,
            "name_grams": char3grams(core_n), "address_grams": char3grams(str_a)
        }
        s1_ids_list.append(sid)

        if old_n: exact_n_lookup[(c, old_n)].append(i)
        if core_n: core_n_lookup[(c, core_n)].append(i)
        if str_a: str_a_lookup[(c, str_a)].append(i)

    del s1; gc.collect()

    # Pass 1B: Scan targets to find rare tokens/grams
    def scan_for_rare(path):
        for chunk in pd.read_csv(path, sep="\t", chunksize=TARGET_CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c = row.country
                for t in tokens(normalize_text(row.business_name)): name_counts[c][t] += 1
                for t in tokens(normalize_text(row.business_address)): addr_counts[c][t] += 1
                for g in char3grams(strong_name_core(row.business_name)): name_gram_counts[c][g] += 1
                for g in char3grams(strong_address(row.business_address)): addr_gram_counts[c][g] += 1
            gc.collect()

    scan_for_rare("dataset/train/train_source2.tsv")
    scan_for_rare("dataset/train/train_source3.tsv")

    for i, sid in enumerate(s1_ids_list):
        row = s1_map[sid]
        c = row["country"]
        for t in row["old_name_tokens"]:
            if 0 < name_counts[c].get(t, 0) <= 100: rare_n_lookup[(c, t)].append(i)
        for t in row["old_address_tokens"]:
            if 0 < addr_counts[c].get(t, 0) <= 100: rare_a_lookup[(c, t)].append(i)
        n_cands = [(name_gram_counts[c].get(g, 0), g) for g in row["name_grams"] if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        if n_cands: n_gram_lookup[(c, sorted(n_cands)[0][1])].append(i)
        a_cands = [(addr_gram_counts[c].get(g, 0), g) for g in row["address_grams"] if 0 < addr_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        if a_cands: a_gram_lookup[(c, sorted(a_cands)[0][1])].append(i)

    print(f"[{get_ram_gb():.2f} GB] PASS 2: Blocking Candidate Generation...")
    heaps = [[] for _ in range(len(s1_ids_list))]
    
    def process_blocking(path):
        for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], chunksize=TARGET_CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c, tid_str = row.country, row.entity_id
                old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
                core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)

                scores = {}
                def touch(idx, amt): scores[idx] = scores.get(idx, 0.0) + amt

                for idx in exact_n_lookup.get((c, old_n), []): touch(idx, 100.0)
                for idx in core_n_lookup.get((c, core_n), []): touch(idx, 80.0)
                for idx in str_a_lookup.get((c, str_a), []): touch(idx, 60.0)
                
                for t in tokens(old_n):
                    f = name_counts[c].get(t, 0)
                    if 0 < f <= 100:
                        w = 15.0 / (1.0 + f**0.5)
                        for idx in rare_n_lookup.get((c, t), []): touch(idx, w)
                        
                for t in tokens(old_a):
                    f = addr_counts[c].get(t, 0)
                    if 0 < f <= 100:
                        w = 20.0 / (1.0 + f**0.5)
                        for idx in rare_a_lookup.get((c, t), []): touch(idx, w)
                        
                for g in char3grams(core_n):
                    f = name_gram_counts[c].get(g, 0)
                    if 0 < f <= GRAM_MAX_FREQ:
                        w = 40.0 / (1.0 + f**0.5)
                        for idx in n_gram_lookup.get((c, g), []): touch(idx, w)
                        
                for g in char3grams(str_a):
                    f = addr_gram_counts[c].get(g, 0)
                    if 0 < f <= GRAM_MAX_FREQ:
                        w = 50.0 / (1.0 + f**0.5)
                        for idx in a_gram_lookup.get((c, g), []): touch(idx, w)

                for idx, score in scores.items():
                    heap = heaps[idx]
                    if len(heap) < MAX_CANDIDATES: heapq.heappush(heap, (score, tid_str))
                    elif score > heap[0][0]: heapq.heapreplace(heap, (score, tid_str))
            gc.collect()

    process_blocking("dataset/train/train_source2.tsv")
    process_blocking("dataset/train/train_source3.tsv")

    print(f"[{get_ram_gb():.2f} GB] PASS 3: Fetching Target Strings...")
    unique_tids = {tid_str for heap in heaps for _, tid_str in heap}
    retained_strings = {}

    def fetch_strings(path):
        for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=TARGET_CHUNK_SIZE):
            chunk = chunk[chunk["entity_id"].isin(unique_tids)]
            for row in chunk.itertuples(index=False):
                retained_strings[row.entity_id] = (normalize_text(row.business_name), normalize_text(row.business_address))
            gc.collect()

    fetch_strings("dataset/train/train_source2.tsv")
    fetch_strings("dataset/train/train_source3.tsv")

    print(f"[{get_ram_gb():.2f} GB] Building Features Array & Predicting...")
    total_cands = sum(len(h) for h in heaps)
    X = np.zeros((total_cands, 9), dtype=np.float32)
    offsets = np.cumsum([0] + [len(h) for h in heaps])[:-1]

    def build_features(idx):
        sid = s1_ids_list[idx]
        selected = heaps[idx]
        offset = offsets[idx]
        s1_n, s1_a = s1_map[sid]["old_name"], s1_map[sid]["old_addr"]

        for i, (block_score, tid_str) in enumerate(selected):
            tgt_n, tgt_a = retained_strings.get(tid_str, ("", ""))

            n_t_set = fuzz.token_set_ratio(s1_n, tgt_n) / 100.0
            n_t_sort = fuzz.token_sort_ratio(s1_n, tgt_n) / 100.0
            n_ratio = fuzz.ratio(s1_n, tgt_n) / 100.0

            a_t_set = fuzz.token_set_ratio(s1_a, tgt_a) / 100.0
            a_t_sort = fuzz.token_sort_ratio(s1_a, tgt_a) / 100.0
            a_ratio = fuzz.ratio(s1_a, tgt_a) / 100.0

            len_n = min(len(s1_n), len(tgt_n)) / (max(len(s1_n), len(tgt_n)) + 1e-5)
            len_a = min(len(s1_a), len(tgt_a)) / (max(len(s1_a), len(tgt_a)) + 1e-5)

            X[offset + i, :] = [n_t_set, n_t_sort, n_ratio, a_t_set, a_t_sort, a_ratio, len_n, len_a, block_score]

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(build_features, range(len(s1_ids_list))))

    probs = model.predict(X)

    print("\n" + "=" * 80)
    print("LIGHTGBM VALIDATION THRESHOLD SWEEP (FROZEN 110k SPLIT)")
    print("=" * 80)

    best_macro_f05 = -1.0
    best_thresh = None
    total_s1_count = len(true_count_by_s1)

    for thresh in THRESHOLDS:
        total_f05 = 0.0
        total_pred, total_tp, total_fp = 0, 0, 0
        
        for idx, sid in enumerate(s1_ids_list):
            true_cnt = true_count_by_s1[sid]
            true_tids = true_pairs_by_s1.get(sid, set())
            
            offset = offsets[idx]
            cands = heaps[idx]
            
            pred_cnt = 0
            tp = 0
            for i, (_, tid_str) in enumerate(cands):
                if probs[offset + i] >= thresh:
                    pred_cnt += 1
                    if tid_str in true_tids: tp += 1
                        
            total_pred += pred_cnt
            total_tp += tp
            total_fp += (pred_cnt - tp)
            total_f05 += entity_f05(true_cnt, pred_cnt, tp)

        macro_f05 = total_f05 / total_s1_count
        print(f"Threshold {thresh:.2f} | Macro F0.5: {macro_f05:.8f} | Predictions: {total_pred:,} | TP: {total_tp:,} | FP: {total_fp:,}")

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_thresh = thresh

    print("\n=================================================")
    print(f"BEST LGBM MACRO F0.5:     {best_macro_f05:.8f} at Threshold {best_thresh:.2f}")
    print(f"Baseline V4 Macro F0.5:   0.70626505")
    print(f"Difference vs Baseline:  {best_macro_f05 - 0.70626505:+.8f}")
    print(f"Total Runtime:            {(time.time() - total_start)/60:.2f} mins")
    print("=================================================")