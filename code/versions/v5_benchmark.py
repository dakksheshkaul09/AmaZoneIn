import pandas as pd
import re
import unicodedata
import gc
import heapq
import time
import psutil
import os

from collections import Counter, defaultdict
from rapidfuzz import fuzz

# =========================================================
# SETTINGS
# =========================================================
CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 200
MATCH_THRESHOLD = 0.86
NAME_WEIGHT = 0.40
ADDRESS_WEIGHT = 0.60
BUDGETS = [50, 100, 200]

start_time = time.time()

def get_ram_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# =========================================================
# 1. NORMALIZATION & GRAMS (Identical to V3/V4)
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

def entity_f05(true_count, predicted_count, tp):
    if true_count == 0: return 1.0 if predicted_count == 0 else 0.0
    if predicted_count == 0: return 0.0
    precision = tp / predicted_count
    recall = tp / true_count
    if precision == 0 or recall == 0: return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

# =========================================================
# 2. LOAD S1 & GROUND TRUTH
# =========================================================
print(f"[{get_ram_gb():.2f} GB] Loading S1 Validation Split...")
val_ids = set(pd.read_csv("dataset/train/validation_s1_ids.tsv", sep="\t")["source1_entity_id"])

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
s1 = s1[s1["entity_id"].isin(val_ids)]

s1_map = {}
for row in s1.itertuples(index=False):
    old_name, old_address = normalize_text(row.business_name), normalize_text(row.business_address)
    core_name, strong_addr = strong_name_core(row.business_name), strong_address(row.business_address)
    s1_map[row.entity_id] = {
        "country": row.country, "old_name": old_name, "old_address": old_address,
        "old_name_tokens": tokens(old_name), "old_address_tokens": tokens(old_address),
        "core_name": core_name, "strong_address": strong_addr,
        "name_grams": char3grams(core_name), "address_grams": char3grams(strong_addr),
    }
del s1; gc.collect()

print(f"[{get_ram_gb():.2f} GB] Loading Ground Truth...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
gt = gt[gt["source1_entity_id"].isin(s1_map.keys())]

true_count_by_s1 = {sid: 0 for sid in s1_map}
target_to_s1 = {}
for row in gt.itertuples(index=False):
    if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids: continue
    ids = row.matched_entity_ids.split(",")
    true_count_by_s1[row.source1_entity_id] = len(ids)
    for tid in ids: target_to_s1[tid] = row.source1_entity_id
del gt; gc.collect()

# =========================================================
# 3. PASS 1: TOKEN & GRAM COUNTS
# =========================================================
print(f"\n[{get_ram_gb():.2f} GB] PASS 1: Counting Tokens & Grams...")
name_counts, address_counts = defaultdict(Counter), defaultdict(Counter)
name_gram_counts, address_gram_counts = defaultdict(Counter), defaultdict(Counter)

def count_stats(path):
    for chunk in pd.read_csv(path, sep="\t", usecols=["business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            country = row.country
            for t in tokens(normalize_text(row.business_name)): name_counts[country][t] += 1
            for t in tokens(normalize_text(row.business_address)): address_counts[country][t] += 1
            for g in char3grams(strong_name_core(row.business_name)): name_gram_counts[country][g] += 1
            for g in char3grams(strong_address(row.business_address)): address_gram_counts[country][g] += 1
        gc.collect()

count_stats("dataset/train/train_source2.tsv")
count_stats("dataset/train/train_source3.tsv")

# Build Reverse Lookups (Identical to V3)
exact_name_lookup, core_name_lookup, strong_address_lookup = defaultdict(list), defaultdict(list), defaultdict(list)
rare_name_token_lookup, rare_address_token_lookup = defaultdict(list), defaultdict(list)
name_gram_lookup, address_gram_lookup = defaultdict(list), defaultdict(list)

for sid, row in s1_map.items():
    country = row["country"]
    if row["old_name"]: exact_name_lookup[(country, row["old_name"])].append(sid)
    if row["core_name"]: core_name_lookup[(country, row["core_name"])].append(sid)
    if row["strong_address"]: strong_address_lookup[(country, row["strong_address"])].append(sid)
    
    for t in row["old_name_tokens"]:
        if 0 < name_counts[country].get(t, 0) <= 100: rare_name_token_lookup[(country, t)].append(sid)
    for t in row["old_address_tokens"]:
        if 0 < address_counts[country].get(t, 0) <= 100: rare_address_token_lookup[(country, t)].append(sid)
        
    n_cands = [(name_gram_counts[country].get(g, 0), g) for g in row["name_grams"] if 0 < name_gram_counts[country].get(g, 0) <= GRAM_MAX_FREQ]
    if n_cands: name_gram_lookup[(country, sorted(n_cands)[0][1])].append(sid)
    
    a_cands = [(address_gram_counts[country].get(g, 0), g) for g in row["address_grams"] if 0 < address_gram_counts[country].get(g, 0) <= GRAM_MAX_FREQ]
    if a_cands: address_gram_lookup[(country, sorted(a_cands)[0][1])].append(sid)

# =========================================================
# 4. PASS 2: BLOCKING CANDIDATE GENERATION
# =========================================================
print(f"\n[{get_ram_gb():.2f} GB] PASS 2: Generating Top-200 Candidates by Blocking Score...")
heaps = defaultdict(list)
total_targets_seen = 0

def process_blocking(path):
    global total_targets_seen
    for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            total_targets_seen += 1
            country, tid = row.country, row.entity_id
            old_name, old_addr = normalize_text(row.business_name), normalize_text(row.business_address)
            core_name, strong_addr = strong_name_core(row.business_name), strong_address(row.business_address)
            
            scores = {}
            def touch(sid, amt): scores[sid] = scores.get(sid, 0.0) + amt
            
            for sid in exact_name_lookup.get((country, old_name), []): touch(sid, 100.0)
            for sid in core_name_lookup.get((country, core_name), []): touch(sid, 80.0)
            for sid in strong_address_lookup.get((country, strong_addr), []): touch(sid, 60.0)
            for t in tokens(old_name):
                f = name_counts[country].get(t, 0)
                if 0 < f <= 100:
                    w = 15.0 / (1.0 + f**0.5)
                    for sid in rare_name_token_lookup.get((country, t), []): touch(sid, w)
            for t in tokens(old_addr):
                f = address_counts[country].get(t, 0)
                if 0 < f <= 100:
                    w = 20.0 / (1.0 + f**0.5)
                    for sid in rare_address_token_lookup.get((country, t), []): touch(sid, w)
            for g in char3grams(core_name):
                f = name_gram_counts[country].get(g, 0)
                if 0 < f <= GRAM_MAX_FREQ:
                    w = 40.0 / (1.0 + f**0.5)
                    for sid in name_gram_lookup.get((country, g), []): touch(sid, w)
            for g in char3grams(strong_addr):
                f = address_gram_counts[country].get(g, 0)
                if 0 < f <= GRAM_MAX_FREQ:
                    w = 50.0 / (1.0 + f**0.5)
                    for sid in address_gram_lookup.get((country, g), []): touch(sid, w)
            
            for sid, score in scores.items():
                heap = heaps[sid]
                if len(heap) < MAX_CANDIDATES:
                    heapq.heappush(heap, (score, tid))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, tid))
        gc.collect()

process_blocking("dataset/train/train_source2.tsv")
process_blocking("dataset/train/train_source3.tsv")

# =========================================================
# 5. PASS 3: FETCH TARGET STRINGS FOR RETAINED CANDIDATES
# =========================================================
print(f"\n[{get_ram_gb():.2f} GB] PASS 3: Extracting Target Strings for Fuzzy Matching...")
unique_retained_targets = set()
for heap in heaps.values():
    unique_retained_targets.update([tid for _, tid in heap])

retained_strings = {}
def fetch_strings(path):
    for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address"], chunksize=CHUNK_SIZE):
        chunk = chunk[chunk["entity_id"].isin(unique_retained_targets)]
        for row in chunk.itertuples(index=False):
            retained_strings[row.entity_id] = (normalize_text(row.business_name), normalize_text(row.business_address))
        gc.collect()

fetch_strings("dataset/train/train_source2.tsv")
fetch_strings("dataset/train/train_source3.tsv")

# =========================================================
# 6. STAGE 2: FUZZY EVALUATION ON BUDGETS
# =========================================================
print(f"\n[{get_ram_gb():.2f} GB] Running Fuzzy Matcher on Retained Budgets...")

for budget in BUDGETS:
    total_f05 = 0.0
    total_predictions, total_tp, total_fp = 0, 0, 0
    singleton_fps = 0
    total_fuzzy_comparisons = 0
    recovered_true_pairs = 0

    for sid in s1_map.keys():
        heap = heaps.get(sid, [])
        selected = heapq.nlargest(budget, heap, key=lambda x: x[0]) if len(heap) > budget else sorted(heap, reverse=True)
        
        true_count = true_count_by_s1.get(sid, 0)
        true_target_ids = {tid for tid, tgt_s1 in target_to_s1.items() if tgt_s1 == sid}
        
        predicted_count, tp = 0, 0
        total_fuzzy_comparisons += len(selected)
        
        selected_ids = {tid for _, tid in selected}
        recovered_true_pairs += len(selected_ids & true_target_ids)

        for _, tid in selected:
            tgt_name, tgt_addr = retained_strings.get(tid, ("", ""))
            
            n_score = fuzz.token_set_ratio(s1_map[sid]["old_name"], tgt_name) / 100.0
            a_score = fuzz.token_set_ratio(s1_map[sid]["old_address"], tgt_addr) / 100.0
            match_score = (NAME_WEIGHT * n_score) + (ADDRESS_WEIGHT * a_score)
            
            if match_score >= MATCH_THRESHOLD:
                predicted_count += 1
                total_predictions += 1
                if tid in true_target_ids:
                    tp += 1
                    total_tp += 1
                else:
                    total_fp += 1
                    
        if true_count == 0 and predicted_count > 0:
            singleton_fps += 1
            
        total_f05 += entity_f05(true_count, predicted_count, tp)

    macro_f05 = total_f05 / len(s1_map)
    candidate_recall = recovered_true_pairs / len(target_to_s1) if len(target_to_s1) > 0 else 0

    print(f"\n--- TOP {budget} BY BLOCKING SCORE (Threshold {MATCH_THRESHOLD}) ---")
    print(f"Candidate Recall:         {candidate_recall:.4%}")
    print(f"Total Retained Cands:     {total_fuzzy_comparisons:,}")
    print(f"Fuzzy Comparisons Made:   {total_fuzzy_comparisons:,}")
    print(f"Singleton False Positives:{singleton_fps:,}")
    print(f"Total True Positives:     {total_tp:,}")
    print(f"Total False Positives:    {total_fp:,}")
    print(f"Macro F0.5 Score:         {macro_f05:.8f}")

runtime = (time.time() - start_time) / 60
print(f"\n=================================================")
print(f"BENCHMARK COMPLETE | Peak RAM: {get_ram_gb():.2f} GB | Runtime: {runtime:.2f} mins")
print(f"=================================================")