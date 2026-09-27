import pandas as pd
import re
import unicodedata
import gc
import heapq
import time
import os
import psutil
import concurrent.futures

from collections import Counter, defaultdict
from rapidfuzz import fuzz

# =========================================================
# SETTINGS
# =========================================================
S1_BATCH_SIZE = 250_000
TARGET_CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 200
MATCH_THRESHOLD = 0.86
NAME_WEIGHT = 0.40
ADDRESS_WEIGHT = 0.60
WORKERS = os.cpu_count() or 4

def get_ram_gb(): 
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# =========================================================
# 1. NORMALIZATION & GRAMS
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
# BATCHED PIPELINE EXECUTION
# =========================================================
if __name__ == "__main__":
    total_start = time.time()

    os.makedirs("output", exist_ok=True)
    
    # Initialize output files
    with open("output/candidate_pairs_v10.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
    with open("output/matching_results_v10.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    # ---------------------------------------------------------
    # PHASE 1: VOCABULARY FILTERING (NO S1 RETENTION)
    # ---------------------------------------------------------
    print(f"\n[{get_ram_gb():.2f} GB] PHASE 1: Extracting S1 Vocabulary...")
    valid_name_tokens, valid_addr_tokens = defaultdict(set), defaultdict(set)
    valid_name_grams, valid_addr_grams = defaultdict(set), defaultdict(set)

    for chunk in pd.read_csv("dataset/test/test_source1.tsv", sep="\t", chunksize=TARGET_CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            c = row.country
            old_name, old_addr = normalize_text(row.business_name), normalize_text(row.business_address)
            valid_name_tokens[c].update(tokens(old_name))
            valid_addr_tokens[c].update(tokens(old_addr))
            valid_name_grams[c].update(char3grams(strong_name_core(row.business_name)))
            valid_addr_grams[c].update(char3grams(strong_address(row.business_address)))
        gc.collect()

    # ---------------------------------------------------------
    # PHASE 2: COUNT TARGETS ONLY FOR S1 VOCABULARY
    # ---------------------------------------------------------
    print(f"[{get_ram_gb():.2f} GB] PHASE 2: Counting Filtered Target Tokens...")
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

    count_targets("dataset/test/test_source2.tsv")
    count_targets("dataset/test/test_source3.tsv")

    del valid_name_tokens, valid_addr_tokens, valid_name_grams, valid_addr_grams
    gc.collect()

    # ---------------------------------------------------------
    # PHASE 3: S1 BATCH PROCESSING 
    # ---------------------------------------------------------
    print(f"[{get_ram_gb():.2f} GB] PHASE 3: Processing S1 in {S1_BATCH_SIZE:,} Row Batches...")
    s1_reader = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", chunksize=S1_BATCH_SIZE)

    for batch_idx, s1_chunk in enumerate(s1_reader):
        t_batch = time.time()
        print(f"\n--- Starting S1 Batch {batch_idx + 1} ({len(s1_chunk):,} rows) [RAM: {get_ram_gb():.2f} GB] ---")

        # 3.1 Build Batch Data
        s1_map = {}
        s1_ids_list = []
        exact_name_lookup, core_name_lookup, strong_addr_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
        rare_name_lookup, rare_addr_lookup = defaultdict(list), defaultdict(list)
        name_gram_lookup, addr_gram_lookup = defaultdict(list), defaultdict(list)

        for i, row in enumerate(s1_chunk.itertuples(index=False)):
            sid, c = row.entity_id, row.country
            s1_ids_list.append(sid)

            old_name, old_addr = normalize_text(row.business_name), normalize_text(row.business_address)
            core_name, str_addr = strong_name_core(row.business_name), strong_address(row.business_address)
            
            s1_map[sid] = {"old_name": old_name, "old_addr": old_addr}

            if old_name: exact_name_lookup[(c, old_name)].append(i)
            if core_name: core_name_lookup[(c, core_name)].append(i)
            if str_addr: strong_addr_lookup[(c, str_addr)].append(i)

            for t in tokens(old_name):
                if 0 < name_counts[c].get(t, 0) <= 100: rare_name_lookup[(c, t)].append(i)
            for t in tokens(old_addr):
                if 0 < addr_counts[c].get(t, 0) <= 100: rare_addr_lookup[(c, t)].append(i)

            n_cands = [(name_gram_counts[c].get(g, 0), g) for g in char3grams(core_name) if 0 < name_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            if n_cands: name_gram_lookup[(c, sorted(n_cands)[0][1])].append(i)

            a_cands = [(addr_gram_counts[c].get(g, 0), g) for g in char3grams(str_addr) if 0 < addr_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
            if a_cands: addr_gram_lookup[(c, sorted(a_cands)[0][1])].append(i)

        # 3.2 Blocking using TRUE VIRTUAL INTEGERS
        print(f"  -> Generating Top-200 Candidates...")
        heaps = [[] for _ in range(len(s1_ids_list))]
        row_counter = [0]  # Array used as mutable reference
        
        def process_blocking(path):
            for chunk in pd.read_csv(path, sep="\t", usecols=["business_name", "business_address", "country"], chunksize=TARGET_CHUNK_SIZE):
                for row in chunk.itertuples(index=False):
                    c = row.country
                    tid_int = row_counter[0]
                    row_counter[0] += 1
                    
                    old_n, old_a = normalize_text(row.business_name), normalize_text(row.business_address)
                    core_n, str_a = strong_name_core(row.business_name), strong_address(row.business_address)
                    
                    scores = {}
                    def touch(idx, amt): scores[idx] = scores.get(idx, 0.0) + amt
                    
                    for idx in exact_name_lookup.get((c, old_n), []): touch(idx, 100.0)
                    for idx in core_name_lookup.get((c, core_n), []): touch(idx, 80.0)
                    for idx in strong_addr_lookup.get((c, str_a), []): touch(idx, 60.0)
                    
                    for t in tokens(old_n):
                        f = name_counts[c].get(t, 0)
                        if 0 < f <= 100:
                            w = 15.0 / (1.0 + f**0.5)
                            for idx in rare_name_lookup.get((c, t), []): touch(idx, w)
                            
                    for t in tokens(old_a):
                        f = addr_counts[c].get(t, 0)
                        if 0 < f <= 100:
                            w = 20.0 / (1.0 + f**0.5)
                            for idx in rare_addr_lookup.get((c, t), []): touch(idx, w)
                            
                    for g in char3grams(core_n):
                        f = name_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            w = 40.0 / (1.0 + f**0.5)
                            for idx in name_gram_lookup.get((c, g), []): touch(idx, w)
                            
                    for g in char3grams(str_a):
                        f = addr_gram_counts[c].get(g, 0)
                        if 0 < f <= GRAM_MAX_FREQ:
                            w = 50.0 / (1.0 + f**0.5)
                            for idx in addr_gram_lookup.get((c, g), []): touch(idx, w)
                            
                    for idx, score in scores.items():
                        heap = heaps[idx]
                        if len(heap) < MAX_CANDIDATES: heapq.heappush(heap, (score, tid_int))
                        elif score > heap[0][0]: heapq.heapreplace(heap, (score, tid_int))
                gc.collect()

        row_counter[0] = 0
        process_blocking("dataset/test/test_source2.tsv")
        process_blocking("dataset/test/test_source3.tsv")

        # 3.3 Extract Strings (Lightning Fast O(1) Lookup)
        print(f"  -> Extracting target strings...")
        unique_retained_ints = {tid_int for heap in heaps for _, tid_int in heap}
        
        retained_strings = {}
        retained_original_ids = {}
        
        def fetch_strings(path):
            for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=TARGET_CHUNK_SIZE):
                for row in chunk.itertuples(index=False):
                    if row_counter[0] in unique_retained_ints:
                        retained_strings[row_counter[0]] = (normalize_text(row.business_name), normalize_text(row.business_address))
                        retained_original_ids[row_counter[0]] = row.entity_id
                    row_counter[0] += 1
                gc.collect()

        row_counter[0] = 0
        fetch_strings("dataset/test/test_source2.tsv")
        fetch_strings("dataset/test/test_source3.tsv")

        # 3.4 Fuzzy Match & File Append
        print(f"  -> Fuzzy matching and appending to disk...")
        def evaluate_s1(idx):
            sid = s1_ids_list[idx]
            selected = sorted(heaps[idx], reverse=True)
            
            all_cands, matches = [], []
            s1_n, s1_a = s1_map[sid]["old_name"], s1_map[sid]["old_addr"]
            
            for _, tid_int in selected:
                tgt_n, tgt_a = retained_strings.get(tid_int, ("", ""))
                original_tid = retained_original_ids.get(tid_int, "")
                if not original_tid: continue
                
                all_cands.append(original_tid)
                
                n_score = fuzz.token_set_ratio(s1_n, tgt_n) / 100.0
                a_score = fuzz.token_set_ratio(s1_a, tgt_a) / 100.0
                if (NAME_WEIGHT * n_score) + (ADDRESS_WEIGHT * a_score) >= MATCH_THRESHOLD:
                    matches.append(original_tid)
                    
            return sid, all_cands, matches

        with open("output/candidate_pairs_v10.tsv", "a", encoding="utf-8") as f_cand, \
             open("output/matching_results_v10.tsv", "a", encoding="utf-8") as f_match:
             
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
                for sid, cands, matches in executor.map(evaluate_s1, range(len(s1_ids_list)), chunksize=1000):
                    f_cand.write(f"{sid}\t{','.join(cands)}\n")
                    f_match.write(f"{sid}\t{','.join(matches)}\n")

        # Nuke all batch memory
        del s1_map, s1_ids_list, heaps, unique_retained_ints, retained_strings, retained_original_ids
        del exact_name_lookup, core_name_lookup, strong_addr_lookup, rare_name_lookup, rare_addr_lookup, name_gram_lookup, addr_gram_lookup
        gc.collect()
        
        print(f"  -> Batch complete. Time: {(time.time() - t_batch)/60:.2f} mins")
        
    print(f"\n=================================================")
    print(f"FULL TEST RUN COMPLETE | Total Runtime: {(time.time() - total_start) / 3600:.2f} hrs")
    print(f"Check /output/ for candidate_pairs_v10.tsv and matching_results_v10.tsv")
    print(f"=================================================")