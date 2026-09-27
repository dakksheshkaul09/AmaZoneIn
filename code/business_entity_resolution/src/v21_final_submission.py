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
# SETTINGS & BUDGET
# =========================================================
CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 100
WORKERS = os.cpu_count() or 4
MATCH_THRESHOLD = 0.60
BATCH_SIZE = 200_000

CAND_PATH = "output/sub_v16_candidate_pairs.tsv"
MATCH_PATH = "output/sub_v16_matching_results.tsv"

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

# =========================================================
# FULL TEST SET PIPELINE (BATCHED & STREAMED S1)
# =========================================================
if __name__ == "__main__":
    t_start = time.time()
    os.makedirs("output", exist_ok=True)
    
    print(f"[{get_ram_gb():.2f} GB] Checking Strict Resume Status...")
    processed_count = 0
    
    if os.path.exists(CAND_PATH) and os.path.exists(MATCH_PATH):
        with open(CAND_PATH, "r+") as fc, open(MATCH_PATH, "r+") as fm:
            fc.readline()
            fm.readline()
            last_good_c, last_good_m = fc.tell(), fm.tell()
            
            while True:
                line_c, line_m = fc.readline(), fm.readline()
                if not line_c or not line_m:
                    break
                    
                sid_c, sid_m = line_c.split("\t")[0], line_m.split("\t")[0]
                if sid_c == sid_m:
                    processed_count += 1
                    last_good_c, last_good_m = fc.tell(), fm.tell()
                else:
                    break
                    
            fc.truncate(last_good_c)
            fm.truncate(last_good_m)
            
        if processed_count > 0:
            print(f"Resuming safely: {processed_count:,} perfectly synced S1 entities found on disk.")
    else:
        with open(CAND_PATH, "w") as fc, open(MATCH_PATH, "w") as fm:
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            fm.write("source1_entity_id\tmatched_entity_ids\n")
        print("Starting fresh submission run.")

    print(f"[{get_ram_gb():.2f} GB] PASS 1: Scanning TEST corpus for token frequencies (Global)...")
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

    scan_for_rare("dataset/test/test_source2.tsv")
    scan_for_rare("dataset/test/test_source3.tsv")

    print(f"[{get_ram_gb():.2f} GB] Loading LightGBM Model...")
    model = lgb.Booster(model_file="models/lgb_model_v16.txt")

    print(f"[{get_ram_gb():.2f} GB] Opening S1 Data Stream...")
    # lambda x avoids converting millions of ints into a list for pandas
    skip_logic = (lambda x: x > 0 and x <= processed_count) if processed_count > 0 else None
    s1_stream = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", chunksize=BATCH_SIZE, skiprows=skip_logic)

    batch_idx = 1
    for s1_chunk in s1_stream:
        print(f"\n=================================================")
        print(f"Processing Batch {batch_idx} ({len(s1_chunk):,} S1 entities)")
        print(f"[{get_ram_gb():.2f} GB] Building S1 Map & Batch Lookups...")
        
        s1_map = {}
        exact_n_lookup, core_n_lookup, str_a_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
        rare_n_lookup, rare_a_lookup = defaultdict(list), defaultdict(list)
        n_gram_lookup, a_gram_lookup = defaultdict(list), defaultdict(list)
        
        for row in s1_chunk.itertuples(index=False):
            old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
            core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)
            s1_map[row.entity_id] = {
                "country": row.country, "old_name": old_n, "old_address": old_a,
                "old_name_tokens": tokens(old_n), "old_address_tokens": tokens(old_a),
                "core_name": core_n, "strong_address": str_a,
                "name_grams": char3grams(core_n), "address_grams": char3grams(str_a)
            }
        
        batch_sids = list(s1_map.keys())
        
        for sid in batch_sids:
            data = s1_map[sid]
            c = data["country"]
            if data["old_name"]: exact_n_lookup[(c, data["old_name"])].append(sid)
            if data["core_name"]: core_n_lookup[(c, data["core_name"])].append(sid)
            if data["strong_address"]: str_a_lookup[(c, data["strong_address"])].append(sid)
            for t in data["old_name_tokens"]:
                if 0 < name_counts[c].get(t, 0) <= 100: rare_n_lookup[(c, t)].append(sid)
            for t in data["old_address_tokens"]:
                if 0 < address_counts[c].get(t, 0) <= 100: rare_a_lookup[(c, t)].append(sid)
            n_cands = [(name_gram_counts[c].get(g, 0), g) for g in data["name_grams"] if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            if n_cands: n_gram_lookup[(c, sorted(n_cands)[0][1])].append(sid)
            a_cands = [(address_gram_counts[c].get(g, 0), g) for g in data["address_grams"] if 0 < address_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            if a_cands: a_gram_lookup[(c, sorted(a_cands)[0][1])].append(sid)

        print(f"[{get_ram_gb():.2f} GB] PASS 2: Generating Candidates for Batch...")
        heaps = defaultdict(list)
        
        def process_blocking(path):
            for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
                for row in chunk.itertuples(index=False):
                    c, tid = row.country, row.entity_id
                    old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
                    core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)

                    scores = {}
                    def touch(sid, amt): scores[sid] = scores.get(sid, 0.0) + amt

                    for sid in exact_n_lookup.get((c, old_n), []): touch(sid, 100.0)
                    for sid in core_n_lookup.get((c, core_n), []): touch(sid, 80.0)
                    for sid in str_a_lookup.get((c, str_a), []): touch(sid, 60.0)
                    for t in tokens(old_n):
                        f = name_counts[c].get(t, 0)
                        if 0 < f <= 100:
                            w = 15.0 / (1.0 + f**0.5)
                            for sid in rare_n_lookup.get((c, t), []): touch(sid, w)
                    for t in tokens(old_a):
                        f = address_counts[c].get(t, 0)
                        if 0 < f <= 100:
                            w = 20.0 / (1.0 + f**0.5)
                            for sid in rare_a_lookup.get((c, t), []): touch(sid, w)
                    for g in char3grams(core_n):
                        f = name_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            w = 40.0 / (1.0 + f**0.5)
                            for sid in n_gram_lookup.get((c, g), []): touch(sid, w)
                    for g in char3grams(str_a):
                        f = address_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            w = 50.0 / (1.0 + f**0.5)
                            for sid in a_gram_lookup.get((c, g), []): touch(sid, w)

                    for sid, score in scores.items():
                        heap = heaps[sid]
                        if len(heap) < MAX_CANDIDATES: heapq.heappush(heap, (score, tid))
                        elif score > heap[0][0]: heapq.heapreplace(heap, (score, tid))
                gc.collect()

        process_blocking("dataset/test/test_source2.tsv")
        process_blocking("dataset/test/test_source3.tsv")

        cands_in_batch = sum(len(heaps.get(sid, [])) for sid in batch_sids)
        if cands_in_batch == 0:
            print(f"[{get_ram_gb():.2f} GB] No candidates found in this batch. Writing empty outputs.")
            with open(CAND_PATH, "a") as fc, open(MATCH_PATH, "a") as fm:
                for sid in batch_sids:
                    fc.write(f"{sid}\t\n")
                    fm.write(f"{sid}\t\n")
            
            del s1_chunk, s1_map, exact_n_lookup, core_n_lookup, str_a_lookup, rare_n_lookup, rare_a_lookup, n_gram_lookup, a_gram_lookup, heaps
            gc.collect()
            batch_idx += 1
            continue

        print(f"[{get_ram_gb():.2f} GB] PASS 3: Extracting Target Strings for Batch...")
        unique_tids = {tid for heap in heaps.values() for _, tid in heap}
        retained_strings = {}

        def fetch_strings(path):
            for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=CHUNK_SIZE):
                chunk = chunk[chunk["entity_id"].isin(unique_tids)]
                for row in chunk.itertuples(index=False):
                    retained_strings[row.entity_id] = (normalize_text(row.business_name), normalize_text(row.business_address))
                gc.collect()

        fetch_strings("dataset/test/test_source2.tsv")
        fetch_strings("dataset/test/test_source3.tsv")

        print(f"[{get_ram_gb():.2f} GB] Building Features & Predicting...")
        X_batch = np.zeros((cands_in_batch, 13), dtype=np.float32)
        offsets = np.cumsum([0] + [len(heaps.get(sid, [])) for sid in batch_sids])[:-1]

        def build_features(idx):
            sid = batch_sids[idx]
            selected = sorted(heaps.get(sid, []), reverse=True)
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

                X_batch[offset + i, :] = [
                    n_t_set, n_t_sort, n_ratio, a_t_set, a_t_sort, a_ratio, len_n, len_a, block_score,
                    n_wratio, a_wratio, n_dig_jac, a_dig_jac
                ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            list(executor.map(build_features, range(len(batch_sids))))

        preds = model.predict(X_batch)

        print(f"[{get_ram_gb():.2f} GB] Writing Outputs for Batch...")
        with open(CAND_PATH, "a") as fc, open(MATCH_PATH, "a") as fm:
            for idx, sid in enumerate(batch_sids):
                selected = sorted(heaps.get(sid, []), reverse=True)
                count = len(selected)
                
                if count == 0:
                    fc.write(f"{sid}\t\n")
                    fm.write(f"{sid}\t\n")
                    continue
                    
                offset = offsets[idx]
                s1_preds = preds[offset : offset + count]
                s1_tids = [tid for _, tid in selected]
                matched_tids = [s1_tids[j] for j in range(count) if s1_preds[j] >= MATCH_THRESHOLD]
                
                fc.write(f"{sid}\t{','.join(s1_tids)}\n")
                fm.write(f"{sid}\t{','.join(matched_tids)}\n")

        print(f"[{get_ram_gb():.2f} GB] Deleting Batch Data & GC...")
        del s1_chunk, s1_map, exact_n_lookup, core_n_lookup, str_a_lookup, rare_n_lookup, rare_a_lookup, n_gram_lookup, a_gram_lookup
        del heaps, unique_tids, retained_strings, X_batch, preds
        gc.collect()
        
        batch_idx += 1

    print(f"\n=================================================")
    print("SUBMISSION GENERATION COMPLETE")
    print(f"Saved: {CAND_PATH}")
    print(f"Saved: {MATCH_PATH}")
    print(f"Total Runtime: {(time.time() - t_start)/60:.2f} mins")
    print("=================================================")