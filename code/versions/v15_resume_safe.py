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
S1_BATCH_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 100
THRESHOLD = 0.60
WORKERS = os.cpu_count() or 4

# --- RESUME SETTING ---
START_BATCH = 2  # Skips Batch 1 and starts directly at Batch 2

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

def tokens(x):
    if not x: return set()
    return set(re.findall(r"\w+", x))

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
# PIPELINE EXECUTION
# =========================================================
if __name__ == "__main__":
    total_start = time.time()
    os.makedirs("output", exist_ok=True)
    
    cand_file = "output/candidate_pairs.tsv"
    match_file = "output/matching_results.tsv"
    
    # SAFETY CHECK: Only write headers and overwrite if starting from scratch
    if START_BATCH == 1:
        with open(cand_file, "w") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(match_file, "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
    else:
        print(f"\n[INFO] Resuming from Batch {START_BATCH}. Output files will be appended.")

    print(f"[{get_ram_gb():.2f} GB] Loading Model...")
    model = lgb.Booster(model_file="models/lgb_model.txt")

    print(f"[{get_ram_gb():.2f} GB] GLOBAL PASS 1: Counting Test Target Tokens & Grams...")
    name_counts = defaultdict(Counter)
    addr_counts = defaultdict(Counter)
    name_gram_counts = defaultdict(Counter)
    addr_gram_counts = defaultdict(Counter)

    def scan_for_rare(path):
        for chunk in pd.read_csv(path, sep="\t", chunksize=TARGET_CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c = row.country
                for t in tokens(normalize_text(row.business_name)): name_counts[c][t] += 1
                for t in tokens(normalize_text(row.business_address)): addr_counts[c][t] += 1
                for g in char3grams(strong_name_core(row.business_name)): name_gram_counts[c][g] += 1
                for g in char3grams(strong_address(row.business_address)): addr_gram_counts[c][g] += 1
            gc.collect()

    scan_for_rare("dataset/test/test_source2.tsv")
    scan_for_rare("dataset/test/test_source3.tsv")

    print(f"[{get_ram_gb():.2f} GB] Initializing S1 Test Stream...")
    s1_iterator = pd.read_csv(
        "dataset/test/test_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        chunksize=S1_BATCH_SIZE
    )
    
    batch_num = 1
    
    for s1_chunk in s1_iterator:
        if batch_num < START_BATCH:
            print(f"--- Skipping S1 Batch {batch_num} (Already Completed) ---")
            batch_num += 1
            continue

        batch_start = time.time()
        print(f"\n--- Starting S1 Batch {batch_num} ({len(s1_chunk):,} rows) [RAM: {get_ram_gb():.2f} GB] ---")
        
        s1_map = {}
        s1_ids_list = []
        exact_n_lookup = defaultdict(list)
        core_n_lookup = defaultdict(list)
        str_a_lookup = defaultdict(list)
        rare_n_lookup = defaultdict(list)
        rare_a_lookup = defaultdict(list)
        n_gram_lookup = defaultdict(list)
        a_gram_lookup = defaultdict(list)

        # 1. Build Index for this batch
        for i, row in enumerate(s1_chunk.itertuples(index=False)):
            c, sid = row.country, row.entity_id
            old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
            core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)

            s1_map[sid] = {
                "country": c, "old_name": old_n, "old_addr": old_a,
                "old_name_tokens": tokens(old_n), "old_address_tokens": tokens(old_a),
                "core_name": core_n, "strong_address": str_a,
                "name_grams": char3grams(core_n), "address_grams": char3grams(str_a)
            }
            s1_ids_list.append(sid)

            if old_n: exact_n_lookup[(c, old_n)].append(i)
            if core_n: core_n_lookup[(c, core_n)].append(i)
            if str_a: str_a_lookup[(c, str_a)].append(i)

            for t in tokens(old_n):
                if 0 < name_counts[c].get(t, 0) <= 100: rare_n_lookup[(c, t)].append(i)
            for t in tokens(old_a):
                if 0 < addr_counts[c].get(t, 0) <= 100: rare_a_lookup[(c, t)].append(i)
                
            n_cands = [(name_gram_counts[c].get(g, 0), g) for g in char3grams(core_n) if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            if n_cands: n_gram_lookup[(c, sorted(n_cands)[0][1])].append(i)
            
            a_cands = [(addr_gram_counts[c].get(g, 0), g) for g in char3grams(str_a) if 0 < addr_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            if a_cands: a_gram_lookup[(c, sorted(a_cands)[0][1])].append(i)

        # 2. Block Candidates
        print(f"  -> Generating Top-100 Candidates...")
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
                            for idx in rare_n_lookup.get((c, t), []): touch(idx, 15.0 / (1.0 + f**0.5))
                    for t in tokens(old_a):
                        f = addr_counts[c].get(t, 0)
                        if 0 < f <= 100:
                            for idx in rare_a_lookup.get((c, t), []): touch(idx, 20.0 / (1.0 + f**0.5))
                    for g in char3grams(core_n):
                        f = name_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            for idx in n_gram_lookup.get((c, g), []): touch(idx, 40.0 / (1.0 + f**0.5))
                    for g in char3grams(str_a):
                        f = addr_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            for idx in a_gram_lookup.get((c, g), []): touch(idx, 50.0 / (1.0 + f**0.5))

                    for idx, score in scores.items():
                        heap = heaps[idx]
                        if len(heap) < MAX_CANDIDATES: heapq.heappush(heap, (score, tid_str))
                        elif score > heap[0][0]: heapq.heapreplace(heap, (score, tid_str))
                gc.collect()

        process_blocking("dataset/test/test_source2.tsv")
        process_blocking("dataset/test/test_source3.tsv")

        # 3. Fetch Strings
        print(f"  -> Extracting target strings...")
        unique_tids = {tid_str for heap in heaps for _, tid_str in heap}
        retained_strings = {}

        def fetch_strings(path):
            for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=TARGET_CHUNK_SIZE):
                chunk = chunk[chunk["entity_id"].isin(unique_tids)]
                for row in chunk.itertuples(index=False):
                    retained_strings[row.entity_id] = (normalize_text(row.business_name), normalize_text(row.business_address))
                gc.collect()

        fetch_strings("dataset/test/test_source2.tsv")
        fetch_strings("dataset/test/test_source3.tsv")

        # 4. Predict & Write
        print(f"  -> Predicting & writing outputs...")
        total_cands = sum(len(h) for h in heaps)
        
        if total_cands > 0:
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
        else:
            probs = np.array([])
            offsets = [0] * len(s1_ids_list)

        with open(cand_file, "a") as f_cand, open(match_file, "a") as f_match:
            for idx, sid in enumerate(s1_ids_list):
                cands = heaps[idx]
                offset = offsets[idx]
                
                all_cand_ids = [tid for _, tid in cands]
                f_cand.write(f"{sid}\t{','.join(all_cand_ids)}\n")
                
                matched_ids = [tid for i, (_, tid) in enumerate(cands) if probs[offset + i] >= THRESHOLD]
                f_match.write(f"{sid}\t{','.join(matched_ids)}\n")

        print(f"  -> Batch complete. Time: {(time.time() - batch_start)/60:.2f} mins")
        batch_num += 1
        
        # 5. EXPLICIT MEMORY CLEANUP
        print(f"  -> Forcing garbage collection...")
        
        # Clear dictionary contents
        s1_map.clear()
        retained_strings.clear()
        exact_n_lookup.clear()
        core_n_lookup.clear()
        str_a_lookup.clear()
        rare_n_lookup.clear()
        rare_a_lookup.clear()
        n_gram_lookup.clear()
        a_gram_lookup.clear()
        
        # Clear lists
        for h in heaps: 
            h.clear()
        
        # Delete variable references entirely
        del s1_map
        del s1_ids_list
        del heaps
        del retained_strings
        del unique_tids
        
        # Safely delete locals that might not exist if batch was empty
        if 'X' in locals(): del X
        if 'probs' in locals(): del probs
        if 'offsets' in locals(): del offsets
        
        # Force Python's GC
        gc.collect()

    print("\n=================================================")
    print(f"FULL TEST RUN COMPLETE | Total Runtime: {(time.time() - total_start)/3600:.2f} hrs")
    print("Files ready for submission:")
    print(f"1. {cand_file}")
    print(f"2. {match_file}")
    print("=================================================")