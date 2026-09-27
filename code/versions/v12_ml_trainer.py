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
TRAIN_S1_SIZE = 50_000
TARGET_CHUNK_SIZE = 250_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 100
WORKERS = os.cpu_count() or 4

def get_ram_gb(): 
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# =========================================================
# 1. NORMALIZATION
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
    "court": "court", "cir": "circle", "circle": "circle", "pl": "place", "place": "place", 
    "ter": "terrace", "terrace": "terrace", "trl": "trail", "trail": "trail", 
    "apt": "apartment", "appt": "apartment", "apartment": "apartment", "fl": "floor", 
    "floor": "floor", "rm": "room", "room": "room", "ste": "suite", "suite": "suite", 
    "bldg": "building", "building": "building", "no": "number", "num": "number", "number": "number"
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

# =========================================================
# TRAINER EXECUTION
# =========================================================
if __name__ == "__main__":
    print(f"[{get_ram_gb():.2f} GB] === STARTING V12 ML TRAINING PIPELINE ===")
    
    # ---------------------------------------------------------
    # 1. LOAD STRICT TRAIN SPLIT (NO LEAKAGE)
    # ---------------------------------------------------------
    val_ids = set(pd.read_csv("dataset/train/validation_s1_ids.tsv", sep="\t")["source1_entity_id"])
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1 = s1[~s1["entity_id"].isin(val_ids)]
    s1 = s1.sample(n=TRAIN_S1_SIZE, random_state=42).copy()

    s1_map, s1_ids_list = {}, []
    valid_name_tokens, valid_addr_tokens = defaultdict(set), defaultdict(set)
    valid_name_grams, valid_addr_grams = defaultdict(set), defaultdict(set)
    
    for i, row in enumerate(s1.itertuples(index=False)):
        c, sid = row.country, row.entity_id
        old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
        core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)
        
        s1_map[sid] = {
            "index": i, "country": c, "old_name": old_n, "old_addr": old_a,
            "old_name_tokens": tokens(old_n), "old_address_tokens": tokens(old_a),
            "core_name": core_n, "strong_address": str_a,
            "name_grams": char3grams(core_n), "address_grams": char3grams(str_a)
        }
        s1_ids_list.append(sid)
        valid_name_tokens[c].update(s1_map[sid]["old_name_tokens"])
        valid_addr_tokens[c].update(s1_map[sid]["old_address_tokens"])
        valid_name_grams[c].update(s1_map[sid]["name_grams"])
        valid_addr_grams[c].update(s1_map[sid]["address_grams"])
        
    del s1; gc.collect()
    
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt["source1_entity_id"].isin(s1_map.keys())]
    true_pairs = set()
    for row in gt.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids: continue
        for tid in row.matched_entity_ids.split(","): true_pairs.add((row.source1_entity_id, tid))
    del gt; gc.collect()

    # ---------------------------------------------------------
    # 2. FILTERED COUNTING
    # ---------------------------------------------------------
    print(f"\n[{get_ram_gb():.2f} GB] Counting Filtered Target Tokens...")
    name_counts, addr_counts = defaultdict(Counter), defaultdict(Counter)
    name_gram_counts, addr_gram_counts = defaultdict(Counter), defaultdict(Counter)
    
    def count_targets(path):
        for chunk in pd.read_csv(path, sep="\t", chunksize=TARGET_CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c = row.country
                for t in tokens(normalize_text(row.business_name)):
                    if t in valid_name_tokens[c]: name_counts[c][t] += 1
                for t in tokens(normalize_text(row.business_address)):
                    if t in valid_addr_tokens[c]: addr_counts[c][t] += 1
                for g in char3grams(strong_name_core(row.business_name)):
                    if g in valid_name_grams[c]: name_gram_counts[c][g] += 1
                for g in char3grams(strong_address(row.business_address)):
                    if g in valid_addr_grams[c]: addr_gram_counts[c][g] += 1
            gc.collect()

    count_targets("dataset/train/train_source2.tsv")
    count_targets("dataset/train/train_source3.tsv")
    del valid_name_tokens, valid_addr_tokens, valid_name_grams, valid_addr_grams; gc.collect()

    exact_n_lookup, core_n_lookup, str_a_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
    rare_n_lookup, rare_a_lookup = defaultdict(list), defaultdict(list)
    n_gram_lookup, a_gram_lookup = defaultdict(list), defaultdict(list)

    for sid, data in s1_map.items():
        c, idx = data["country"], data["index"]
        if data["old_name"]: exact_n_lookup[(c, data["old_name"])].append(idx)
        if data["core_name"]: core_n_lookup[(c, data["core_name"])].append(idx)
        if data["strong_address"]: str_a_lookup[(c, data["strong_address"])].append(idx)
        
        for t in data["old_name_tokens"]:
            if 0 < name_counts[c].get(t, 0) <= 100: rare_n_lookup[(c, t)].append(idx)
        for t in data["old_address_tokens"]:
            if 0 < addr_counts[c].get(t, 0) <= 100: rare_a_lookup[(c, t)].append(idx)

        n_cands = [(name_gram_counts[c].get(g, 0), g) for g in data["name_grams"] if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        if n_cands: n_gram_lookup[(c, sorted(n_cands)[0][1])].append(idx)
        
        a_cands = [(addr_gram_counts[c].get(g, 0), g) for g in data["address_grams"] if 0 < addr_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        if a_cands: a_gram_lookup[(c, sorted(a_cands)[0][1])].append(idx)

    # ---------------------------------------------------------
    # 3. BLOCKING (MINE HARD NEGATIVES WITH RAW STRINGS)
    # ---------------------------------------------------------
    print(f"\n[{get_ram_gb():.2f} GB] Generating Candidates (Mining Pairs)...")
    heaps = [[] for _ in range(len(s1_map))]
    
    def process_blocking(path):
        processed = 0
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
            processed += len(chunk)
            print(f"    ... Scanned {processed:,} targets")
            gc.collect()

    process_blocking("dataset/train/train_source2.tsv")
    process_blocking("dataset/train/train_source3.tsv")

    # ---------------------------------------------------------
    # 4. EXTRACT TARGET STRINGS
    # ---------------------------------------------------------
    print(f"\n[{get_ram_gb():.2f} GB] Extracting Candidate Strings...")
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

    # ---------------------------------------------------------
    # 5. FEATURE ENGINEERING (X, y)
    # ---------------------------------------------------------
    print(f"\n[{get_ram_gb():.2f} GB] Computing String Features via {WORKERS} Thread Workers...")
    
    def build_features(idx):
        sid = s1_ids_list[idx]
        selected = heaps[idx]
        s1_n, s1_a = s1_map[sid]["old_name"], s1_map[sid]["old_addr"]
        
        batch_X, batch_y = [], []
        
        for block_score, tid_str in selected:
            tgt_n, tgt_a = retained_strings.get(tid_str, ("", ""))
            
            n_t_set = fuzz.token_set_ratio(s1_n, tgt_n) / 100.0
            n_t_sort = fuzz.token_sort_ratio(s1_n, tgt_n) / 100.0
            n_ratio = fuzz.ratio(s1_n, tgt_n) / 100.0
            
            a_t_set = fuzz.token_set_ratio(s1_a, tgt_a) / 100.0
            a_t_sort = fuzz.token_sort_ratio(s1_a, tgt_a) / 100.0
            a_ratio = fuzz.ratio(s1_a, tgt_a) / 100.0
            
            len_n = min(len(s1_n), len(tgt_n)) / (max(len(s1_n), len(tgt_n)) + 1e-5)
            len_a = min(len(s1_a), len(tgt_a)) / (max(len(s1_a), len(tgt_a)) + 1e-5)
            
            batch_X.append([n_t_set, n_t_sort, n_ratio, a_t_set, a_t_sort, a_ratio, len_n, len_a, block_score])
            batch_y.append(1 if (sid, tid_str) in true_pairs else 0)
            
        return batch_X, batch_y

    X, y = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for res_X, res_y in executor.map(build_features, range(len(s1_ids_list))):
            X.extend(res_X)
            y.extend(res_y)

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

    print(f"Generated {len(y):,} training pairs.")
    print(f"True Matches (1): {y.sum():,} | Hard Negatives (0): {len(y) - y.sum():,}")

    del s1_map, heaps, retained_strings, exact_n_lookup, core_n_lookup
    gc.collect()

    # ---------------------------------------------------------
    # 6. TRAIN LIGHTGBM
    # ---------------------------------------------------------
    print(f"\n[{get_ram_gb():.2f} GB] Training LightGBM Classifier...")
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=300)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/lgb_model.txt")
    print("\nModel saved to models/lgb_model.txt")
    
    print("\n=================================================")
    print(f"TRAINING COMPLETE | Peak RAM: {get_ram_gb():.2f} GB")
    print("=================================================")