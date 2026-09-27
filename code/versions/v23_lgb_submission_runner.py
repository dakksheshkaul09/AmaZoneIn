import os
import pandas as pd
import numpy as np
import re
import unicodedata
import gc
import heapq
import time
import psutil
import concurrent.futures
import lightgbm as lgb
from collections import Counter, defaultdict
from rapidfuzz import fuzz

# =========================================================
# CONFIGURATION & HYPERPARAMETERS
# =========================================================
CHUNK_SIZE = 250_000
S1_CHUNK_SIZE = 200_000  # Safely tuned to 200k for 16 GB RAM limits
GRAM_MAX_FREQ = 5000
TOKEN_MAX_FREQ = 5000  # V23A Relaxed Token Limit
MAX_CANDIDATES = 100
MATCH_THRESHOLD = 0.60  # V16 LightGBM Probability Threshold
WORKERS = os.cpu_count() or 4

# Paths
S2_PATH = "dataset/test/test_source2.tsv"
S3_PATH = "dataset/test/test_source3.tsv"
S1_PATH = "dataset/test/test_source1.tsv"
MODEL_PATH = "models/lgb_model_v16.txt"

CANDIDATES_OUT = "output/v23_lgb_candidate_pairs.tsv"
MATCHES_OUT = "output/v23_lgb_matching_results.tsv"

os.makedirs("output", exist_ok=True)

def get_ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# =========================================================
# NORMALIZATION & FEATURE UTILS
# =========================================================
LEGAL_SUFFIX_MAP = {
    "pvt": "private", "private": "private", "ltd": "limited", "limited": "limited",
    "corp": "corporation", "corporation": "corporation", "inc": "incorporated", 
    "incorporated": "incorporated", "co": "company", "company": "company",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp"
}
LEGAL_SUFFIXES = set(LEGAL_SUFFIX_MAP.values())

ADDRESS_MAP = {
    "rd": "road", "road": "road", "st": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "dr": "drive", "drive": "drive", "ln": "lane", "lane": "lane",
    "blvd": "boulevard", "boulevard": "boulevard", "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway", "ct": "court", "court": "court",
    "cir": "circle", "circle": "circle", "pl": "place", "place": "place",
    "ter": "terrace", "terrace": "terrace", "trl": "trail", "trail": "trail",
    "apt": "apartment", "appt": "apartment", "apartment": "apartment",
    "fl": "floor", "floor": "floor", "rm": "room", "room": "room",
    "ste": "suite", "suite": "suite", "bldg": "building", "building": "building",
    "no": "number", "num": "number", "number": "number"
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
# PRODUCTION PIPELINE WITH HARDENED RESUME & LIGHTGBM
# =========================================================
if __name__ == "__main__":
    total_start = time.time()
    
    print(f"[{get_ram_gb():.2f} GB] Checking Hardened Synchronized Resume Status...")
    processed_sids = set()
    
    cand_exists = os.path.exists(CANDIDATES_OUT)
    match_exists = os.path.exists(MATCHES_OUT)

    if cand_exists != match_exists:
        raise RuntimeError(
            "Resume safety check failed: exactly one output file exists. "
            "Delete or recover the incomplete outputs manually before restarting."
        )

    if cand_exists and match_exists:
        with open(CANDIDATES_OUT, "r+") as fc, open(MATCHES_OUT, "r+") as fm:
            fc.readline()
            fm.readline()
            last_good_c, last_good_m = fc.tell(), fm.tell()
            
            while True:
                line_c, line_m = fc.readline(), fm.readline()
                if not line_c or not line_m:
                    break
                # Only accept fully newline-terminated records to avoid partial crash artifacts
                if not line_c.endswith("\n") or not line_m.endswith("\n"):
                    break
                    
                sid_c = line_c.split("\t", 1)[0]
                sid_m = line_m.split("\t", 1)[0]
                
                if sid_c == sid_m:
                    processed_sids.add(sid_c)
                    last_good_c, last_good_m = fc.tell(), fm.tell()
                else:
                    break
            # Truncate any incomplete or desynced trailing lines
            fc.truncate(last_good_c)
            fm.truncate(last_good_m)
            
        if processed_sids:
            print(f"Resuming safely: Found {len(processed_sids):,} perfectly synced S1 entities.")
    else:
        with open(CANDIDATES_OUT, "w") as fc, open(MATCHES_OUT, "w") as fm:
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            fm.write("source1_entity_id\tmatched_entity_ids\n")

    # ---------------------------------------------------------
    # PASS 1: GLOBAL TEST TARGET PROFILING
    # ---------------------------------------------------------
    t0 = time.time()
    print(f"[{get_ram_gb():.2f} GB] PASS 1: Global Count of Test Tokens & Grams...")
    name_counts, address_counts = defaultdict(Counter), defaultdict(Counter)
    name_gram_counts, address_gram_counts = defaultdict(Counter), defaultdict(Counter)

    def count_stats(path):
        for chunk in pd.read_csv(path, sep="\t", usecols=["business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c = str(row.country)
                for t in tokens(normalize_text(row.business_name)): name_counts[c][t] += 1
                for t in tokens(normalize_text(row.business_address)): address_counts[c][t] += 1
                for g in char3grams(strong_name_core(row.business_name)): name_gram_counts[c][g] += 1
                for g in char3grams(strong_address(row.business_address)): address_gram_counts[c][g] += 1
            gc.collect()

    count_stats(S2_PATH)
    count_stats(S3_PATH)
    print(f"  -> Profiling complete in {(time.time() - t0)/60:.2f} mins")

    print(f"[{get_ram_gb():.2f} GB] Loading LightGBM Model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=MODEL_PATH)

    # ---------------------------------------------------------
    # CHUNKED TEST SET EXECUTION (200k S1 chunks)
    # ---------------------------------------------------------
    s1_iterator = pd.read_csv(S1_PATH, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], chunksize=S1_CHUNK_SIZE)
    
    for chunk_idx, s1_chunk in enumerate(s1_iterator):
        # Filter out already processed S1s for resume safety
        s1_chunk = s1_chunk[~s1_chunk["entity_id"].astype(str).isin(processed_sids)]
        if len(s1_chunk) == 0:
            continue
            
        chunk_start = time.time()
        print(f"\n========================================================")
        print(f"[{get_ram_gb():.2f} GB] STARTING CHUNK {chunk_idx + 1} ({len(s1_chunk):,} S1 Rows)")
        print(f"========================================================")
        
        s1_map = {}
        s1_ids_list = []
        for i, row in enumerate(s1_chunk.itertuples(index=False)):
            old_name, old_addr = normalize_text(row.business_name), normalize_text(row.business_address)
            core_name, strong_addr = strong_name_core(row.business_name), strong_address(row.business_address)
            s1_map[str(row.entity_id)] = {
                "index": i, "country": str(row.country), "old_name": old_name, "old_address": old_addr,
                "old_name_tokens": tokens(old_name), "old_address_tokens": tokens(old_addr),
                "core_name": core_name, "strong_address": strong_addr,
                "name_grams": char3grams(core_name), "address_grams": char3grams(strong_addr),
            }
            s1_ids_list.append(str(row.entity_id))
            
        exact_name_lookup, core_name_lookup, strong_address_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
        rare_name_token_lookup, rare_address_token_lookup = defaultdict(list), defaultdict(list)
        name_gram_lookup, address_gram_lookup = defaultdict(list), defaultdict(list)

        for sid, row in s1_map.items():
            c, idx = row["country"], row["index"]
            if row["old_name"]: exact_name_lookup[(c, row["old_name"])].append(idx)
            if row["core_name"]: core_name_lookup[(c, row["core_name"])].append(idx)
            if row["strong_address"]: strong_address_lookup[(c, row["strong_address"])].append(idx)
            
            for t in row["old_name_tokens"]:
                if 0 < name_counts[c].get(t, 0) <= TOKEN_MAX_FREQ: rare_name_token_lookup[(c, t)].append(idx)
            for t in row["old_address_tokens"]:
                if 0 < address_counts[c].get(t, 0) <= TOKEN_MAX_FREQ: rare_address_token_lookup[(c, t)].append(idx)
                
            n_cands = [(name_gram_counts[c].get(g, 0), g) for g in row["name_grams"] if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            for _, g in sorted(n_cands)[:3]: name_gram_lookup[(c, g)].append(idx)
                
            a_cands = [(address_gram_counts[c].get(g, 0), g) for g in row["address_grams"] if 0 < address_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            for _, g in sorted(a_cands)[:3]: address_gram_lookup[(c, g)].append(idx)

        # PASS 2: GENERATE CANDIDATES FOR CHUNK
        print(f"[{get_ram_gb():.2f} GB] Pass 2: Generating V23A Blocking Heaps...")
        heaps = [[] for _ in range(len(s1_map))]
        global_row_counter = 0

        def process_blocking(path):
            global global_row_counter
            for chunk in pd.read_csv(path, sep="\t", usecols=["business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
                for row in chunk.itertuples(index=False):
                    c, tid_int = str(row.country), global_row_counter
                    global_row_counter += 1
                    
                    old_name, old_addr = normalize_text(row.business_name), normalize_text(row.business_address)
                    core_name, strong_addr = strong_name_core(row.business_name), strong_address(row.business_address)
                    
                    scores = {}
                    def touch(idx, amt): scores[idx] = scores.get(idx, 0.0) + amt
                    
                    for idx in exact_name_lookup.get((c, old_name), []): touch(idx, 100.0)
                    for idx in core_name_lookup.get((c, core_name), []): touch(idx, 80.0)
                    for idx in strong_address_lookup.get((c, strong_addr), []): touch(idx, 60.0)
                    
                    for t in tokens(old_name):
                        f = name_counts[c].get(t, 0)
                        if 0 < f <= TOKEN_MAX_FREQ:
                            w = 15.0 / (1.0 + f**0.5)
                            for idx in rare_name_token_lookup.get((c, t), []): touch(idx, w)
                    
                    for t in tokens(old_addr):
                        f = address_counts[c].get(t, 0)
                        if 0 < f <= TOKEN_MAX_FREQ:
                            w = 20.0 / (1.0 + f**0.5)
                            for idx in rare_address_token_lookup.get((c, t), []): touch(idx, w)
                    
                    for g in char3grams(core_name):
                        f = name_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            w = 40.0 / (1.0 + f**0.5)
                            for idx in name_gram_lookup.get((c, g), []): touch(idx, w)
                    
                    for g in char3grams(strong_addr):
                        f = address_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            w = 50.0 / (1.0 + f**0.5)
                            for idx in address_gram_lookup.get((c, g), []): touch(idx, w)
                    
                    for idx, score in scores.items():
                        heap = heaps[idx]
                        if len(heap) < MAX_CANDIDATES: heapq.heappush(heap, (score, tid_int))
                        elif score > heap[0][0]: heapq.heapreplace(heap, (score, tid_int))
                gc.collect()

        process_blocking(S2_PATH)
        process_blocking(S3_PATH)

        cands_in_chunk = sum(len(h) for h in heaps)
        if cands_in_chunk == 0:
            print(f"[{get_ram_gb():.2f} GB] No candidates found in chunk. Writing empty outputs with full cleanup.")
            with open(CANDIDATES_OUT, "a") as f_cand, open(MATCHES_OUT, "a") as f_match:
                for sid in s1_ids_list:
                    f_cand.write(f"{sid}\t\n")
                    f_match.write(f"{sid}\t\n")
            del s1_map, s1_ids_list, heaps, exact_name_lookup, core_name_lookup, strong_address_lookup
            del rare_name_token_lookup, rare_address_token_lookup, name_gram_lookup, address_gram_lookup
            gc.collect()
            continue

        # PASS 3: FETCH TARGET STRINGS
        print(f"[{get_ram_gb():.2f} GB] Pass 3: Fetching Target Strings...")
        unique_retained_ints = set()
        for heap in heaps: unique_retained_ints.update([tid_int for _, tid_int in heap])

        retained_strings, retained_original_ids = {}, {}
        global_row_counter = 0

        def fetch_strings(path):
            global global_row_counter
            for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=CHUNK_SIZE):
                for row in chunk.itertuples(index=False):
                    if global_row_counter in unique_retained_ints:
                        retained_strings[global_row_counter] = (normalize_text(row.business_name), normalize_text(row.business_address))
                        retained_original_ids[global_row_counter] = str(row.entity_id)
                    global_row_counter += 1
                gc.collect()

        fetch_strings(S2_PATH)
        fetch_strings(S3_PATH)

        # STAGE 2: 13-FEATURE MATRIX & LIGHTGBM INFERENCE
        print(f"[{get_ram_gb():.2f} GB] Stage 2: Building 13 LightGBM Features & Predicting...")
        X_batch = np.zeros((cands_in_chunk, 13), dtype=np.float32)
        offsets = np.cumsum([0] + [len(h) for h in heaps])[:-1]

        def build_features(idx):
            sid = s1_ids_list[idx]
            heap = heaps[idx]
            selected = sorted(heap, reverse=True)
            offset = offsets[idx]
            s1_n, s1_a = s1_map[sid]["old_name"], s1_map[sid]["old_address"]

            for i, (block_score, tid_int) in enumerate(selected):
                tgt_name, tgt_addr = retained_strings.get(tid_int, ("", ""))
                
                n_t_set = fuzz.token_set_ratio(s1_n, tgt_name) / 100.0
                n_t_sort = fuzz.token_sort_ratio(s1_n, tgt_name) / 100.0
                n_ratio = fuzz.ratio(s1_n, tgt_name) / 100.0
                a_t_set = fuzz.token_set_ratio(s1_a, tgt_addr) / 100.0
                a_t_sort = fuzz.token_sort_ratio(s1_a, tgt_addr) / 100.0
                a_ratio = fuzz.ratio(s1_a, tgt_addr) / 100.0
                len_n = min(len(s1_n), len(tgt_name)) / (max(len(s1_n), len(tgt_name)) + 1e-5)
                len_a = min(len(s1_a), len(tgt_addr)) / (max(len(s1_a), len(tgt_addr)) + 1e-5)
                n_wratio = fuzz.WRatio(s1_n, tgt_name) / 100.0
                a_wratio = fuzz.WRatio(s1_a, tgt_addr) / 100.0
                n_dig_jac = digit_jaccard(s1_n, tgt_name)
                a_dig_jac = digit_jaccard(s1_a, tgt_addr)

                X_batch[offset + i, :] = [
                    n_t_set, n_t_sort, n_ratio, a_t_set, a_t_sort, a_ratio, len_n, len_a, block_score,
                    n_wratio, a_wratio, n_dig_jac, a_dig_jac
                ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            list(executor.map(build_features, range(len(s1_ids_list))))

        preds = model.predict(X_batch)

        print(f"[{get_ram_gb():.2f} GB] Writing Outputs for Chunk...")
        with open(CANDIDATES_OUT, "a") as f_cand, open(MATCHES_OUT, "a") as f_match:
            for idx, sid in enumerate(s1_ids_list):
                heap = heaps[idx]
                selected = sorted(heap, reverse=True)
                count = len(selected)
                
                if count == 0:
                    f_cand.write(f"{sid}\t\n")
                    f_match.write(f"{sid}\t\n")
                    continue
                    
                offset = offsets[idx]
                s1_preds = preds[offset : offset + count]
                
                s1_cands, s1_matches = [], []
                for j, (_, tid_int) in enumerate(selected):
                    original_tid = retained_original_ids.get(tid_int, "")
                    if not original_tid: continue
                    s1_cands.append(original_tid)
                    if s1_preds[j] >= MATCH_THRESHOLD:
                        s1_matches.append(original_tid)
                
                f_cand.write(f"{sid}\t{','.join(s1_cands)}\n")
                f_match.write(f"{sid}\t{','.join(s1_matches)}\n")

        print(f"  -> Chunk {chunk_idx + 1} finished in {(time.time() - chunk_start)/60:.2f} mins")
        del s1_map, s1_ids_list, heaps, unique_retained_ints, retained_strings, retained_original_ids, X_batch, preds
        del exact_name_lookup, core_name_lookup, strong_address_lookup, rare_name_token_lookup, rare_address_token_lookup, name_gram_lookup, address_gram_lookup
        gc.collect()

    print(f"\n========================================================")
    print(f"FULL TEST RUN COMPLETE | Total Time: {(time.time() - total_start)/60:.2f} mins")
    print(f"Submission Files Generated Safely:")
    print(f"1. {CANDIDATES_OUT}")
    print(f"2. {MATCHES_OUT}")
    print(f"========================================================")