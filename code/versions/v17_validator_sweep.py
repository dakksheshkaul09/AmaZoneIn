import pandas as pd
import numpy as np
import re
import unicodedata
import gc
import heapq
import time
import os
import psutil
import concurrent.futures
import lightgbm as lgb
from collections import Counter, defaultdict
from rapidfuzz import fuzz

# =========================================================
# SETTINGS
# =========================================================
CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 100  # Production budget
WORKERS = os.cpu_count() or 4
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def get_ram_gb(): 
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# =========================================================
# UTILS & 13 FEATURES
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

def get_digits(text): return set(re.findall(r'\d+', text))

def digit_jaccard(t1, t2):
    d1, d2 = get_digits(t1), get_digits(t2)
    if not d1 and not d2: return -1.0
    if not d1 or not d2: return 0.0
    return len(d1 & d2) / len(d1 | d2)

def entity_f05(true_count, predicted_count, tp):
    if true_count == 0: return 1.0 if predicted_count == 0 else 0.0
    if predicted_count == 0: return 0.0
    precision = tp / predicted_count
    recall = tp / true_count
    if precision == 0 or recall == 0: return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

# =========================================================
# VALIDATOR PIPELINE
# =========================================================
if __name__ == "__main__":
    t0 = time.time()
    
    print(f"[{get_ram_gb():.2f} GB] Loading Validation S1 & Ground Truth...")
    val_ids_df = pd.read_csv("dataset/train/validation_s1_ids.tsv", sep="\t")
    val_ids = set(val_ids_df["source1_entity_id"])

    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt_df = gt_df[gt_df["source1_entity_id"].isin(val_ids)].copy()
    
    true_targets_by_s1 = {}
    for row in gt_df.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids:
            true_targets_by_s1[row.source1_entity_id] = set()
        else:
            true_targets_by_s1[row.source1_entity_id] = set(row.matched_entity_ids.split(","))

    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1 = s1[s1["entity_id"].isin(val_ids)].copy()

    s1_map = {}
    exact_n_lookup, core_n_lookup, str_a_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
    rare_n_lookup, rare_a_lookup = defaultdict(list), defaultdict(list)
    n_gram_lookup, a_gram_lookup = defaultdict(list), defaultdict(list)
    
    for i, row in enumerate(s1.itertuples(index=False)):
        c, sid = row.country, row.entity_id
        old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
        core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)

        s1_map[sid] = {
            "index": i, "country": c, "old_name": old_n, "old_address": old_a,
            "old_name_tokens": tokens(old_n), "old_address_tokens": tokens(old_a),
            "core_name": core_n, "strong_address": str_a,
            "name_grams": char3grams(core_n), "address_grams": char3grams(str_a)
        }
    s1_ids_list = list(s1_map.keys())
    del s1, gt_df; gc.collect()

    print(f"[{get_ram_gb():.2f} GB] PASS 1: Scanning target corpus for token frequencies...")
    name_counts, address_counts = defaultdict(Counter), defaultdict(Counter)
    name_gram_counts, address_gram_counts = defaultdict(Counter), defaultdict(Counter)

    def scan_for_rare(path):
        for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c = row.country
                for t in tokens(normalize_text(row.business_name)): name_counts[c][t] += 1
                for t in tokens(normalize_text(row.business_address)): address_counts[c][t] += 1
                for g in char3grams(strong_name_core(row.business_name)): name_gram_counts[c][g] += 1
                for g in char3grams(strong_address(row.business_address)): address_gram_counts[c][g] += 1
            gc.collect()

    scan_for_rare("dataset/train/train_source2.tsv")
    scan_for_rare("dataset/train/train_source3.tsv")

    # Finalize lookups
    for sid, data in s1_map.items():
        c, i = data["country"], data["index"]
        if data["old_name"]: exact_n_lookup[(c, data["old_name"])].append(i)
        if data["core_name"]: core_n_lookup[(c, data["core_name"])].append(i)
        if data["strong_address"]: str_a_lookup[(c, data["strong_address"])].append(i)
        for t in data["old_name_tokens"]:
            if 0 < name_counts[c].get(t, 0) <= 100: rare_n_lookup[(c, t)].append(i)
        for t in data["old_address_tokens"]:
            if 0 < address_counts[c].get(t, 0) <= 100: rare_a_lookup[(c, t)].append(i)
        n_cands = [(name_gram_counts[c].get(g, 0), g) for g in data["name_grams"] if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        if n_cands: n_gram_lookup[(c, sorted(n_cands)[0][1])].append(i)
        a_cands = [(address_gram_counts[c].get(g, 0), g) for g in data["address_grams"] if 0 < address_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        if a_cands: a_gram_lookup[(c, sorted(a_cands)[0][1])].append(i)

    print(f"\n[{get_ram_gb():.2f} GB] PASS 2: Generating Top-{MAX_CANDIDATES} Candidates...")
    heaps = [[] for _ in range(len(s1_map))]
    
    def process_blocking(path):
        for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c, tid = row.country, row.entity_id
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
                    f = address_counts[c].get(t, 0)
                    if 0 < f <= 100:
                        w = 20.0 / (1.0 + f**0.5)
                        for idx in rare_a_lookup.get((c, t), []): touch(idx, w)
                        
                for g in char3grams(core_n):
                    f = name_gram_counts[c].get(g, 0)
                    if 0 < f <= GRAM_MAX_FREQ:
                        w = 40.0 / (1.0 + f**0.5)
                        for idx in n_gram_lookup.get((c, g), []): touch(idx, w)
                        
                for g in char3grams(str_a):
                    f = address_gram_counts[c].get(g, 0)
                    if 0 < f <= GRAM_MAX_FREQ:
                        w = 50.0 / (1.0 + f**0.5)
                        for idx in a_gram_lookup.get((c, g), []): touch(idx, w)

                for idx, score in scores.items():
                    heap = heaps[idx]
                    if len(heap) < MAX_CANDIDATES: heapq.heappush(heap, (score, tid))
                    elif score > heap[0][0]: heapq.heapreplace(heap, (score, tid))
            gc.collect()

    process_blocking("dataset/train/train_source2.tsv")
    process_blocking("dataset/train/train_source3.tsv")

    print(f"\n[{get_ram_gb():.2f} GB] PASS 3: Extracting Target Strings...")
    unique_tids = {tid for heap in heaps for _, tid in heap}
    retained_strings = {}

    def fetch_strings(path):
        for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=CHUNK_SIZE):
            chunk = chunk[chunk["entity_id"].isin(unique_tids)]
            for row in chunk.itertuples(index=False):
                retained_strings[row.entity_id] = (normalize_text(row.business_name), normalize_text(row.business_address))
            gc.collect()

    fetch_strings("dataset/train/train_source2.tsv")
    fetch_strings("dataset/train/train_source3.tsv")

    print(f"\n[{get_ram_gb():.2f} GB] Building Validation Feature Matrix...")
    total_cands = sum(len(h) for h in heaps)
    X = np.zeros((total_cands, 13), dtype=np.float32)
    offsets = np.cumsum([0] + [len(h) for h in heaps])[:-1]

    def build_features(idx):
        sid = s1_ids_list[idx]
        selected = sorted(heaps[idx], reverse=True)
        offset = offsets[idx]
        s1_n, s1_a = s1_map[sid]["old_name"], s1_map[sid]["old_address"]

        for i, (block_score, tid) in enumerate(selected):
            tgt_n, tgt_a = retained_strings.get(tid, ("", ""))
            
            n_t_set = fuzz.token_set_ratio(s1_n, tgt_n) / 100.0
            n_t_sort = fuzz.token_sort_ratio(s1_n, tgt_n) / 100.0
            n_ratio = fuzz.ratio(s1_n, tgt_n) / 100.0
            a_t_set = fuzz.token_set_ratio(s1_a, tgt_a) / 100.0
            a_t_sort = fuzz.token_sort_ratio(s1_a, tgt_a) / 100.0
            a_ratio = fuzz.ratio(s1_a, tgt_a) / 100.0
            len_n = min(len(s1_n), len(tgt_n)) / (max(len(s1_n), len(tgt_n)) + 1e-5)
            len_a = min(len(s1_a), len(tgt_a)) / (max(len(s1_a), len(tgt_a)) + 1e-5)
            n_wratio = fuzz.WRatio(s1_n, tgt_n) / 100.0
            a_wratio = fuzz.WRatio(s1_a, tgt_a) / 100.0
            n_dig_jac = digit_jaccard(s1_n, tgt_n)
            a_dig_jac = digit_jaccard(s1_a, tgt_a)

            X[offset + i, :] = [
                n_t_set, n_t_sort, n_ratio, a_t_set, a_t_sort, a_ratio, len_n, len_a, block_score,
                n_wratio, a_wratio, n_dig_jac, a_dig_jac
            ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(build_features, range(len(s1_ids_list))))

    print(f"[{get_ram_gb():.2f} GB] Generating Probabilities via LightGBM...")
    model = lgb.Booster(model_file="models/lgb_model_v16.txt")
    preds = model.predict(X)

    print(f"\n=================================================")
    print("V16 13-FEATURE THRESHOLD SWEEP RESULTS")
    print("=================================================")
    best_f05 = -1.0
    best_thresh = None

    for thresh in THRESHOLDS:
        total_f05 = 0.0
        total_tp, total_fp, singleton_fps = 0, 0, 0
        total_preds = 0
        
        for idx in range(len(s1_ids_list)):
            sid = s1_ids_list[idx]
            offset = offsets[idx]
            count = len(heaps[idx])
            
            s1_preds = preds[offset : offset + count]
            s1_tids = [tid for _, tid in sorted(heaps[idx], reverse=True)]
            
            predicted_tids = {s1_tids[j] for j in range(count) if s1_preds[j] >= thresh}
            true_tids = true_targets_by_s1.get(sid, set())
            
            pred_count = len(predicted_tids)
            tp = len(predicted_tids & true_tids)
            
            total_preds += pred_count
            total_tp += tp
            total_fp += (pred_count - tp)
            if len(true_tids) == 0 and pred_count > 0:
                singleton_fps += 1
                
            total_f05 += entity_f05(len(true_tids), pred_count, tp)

        macro_f05 = total_f05 / len(s1_ids_list)
        
        print(f"Threshold: {thresh:.2f} | Macro F0.5: {macro_f05:.6f} | TP: {total_tp:,} | FP: {total_fp:,} | Sing. FPs: {singleton_fps:,}")
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh

    print(f"\n=================================================")
    print(f"BEST THRESHOLD: {best_thresh:.2f} (Macro F0.5 = {best_f05:.6f})")
    print(f"Total Validation Runtime: {(time.time() - t0)/60:.2f} mins")
    print(f"=================================================")