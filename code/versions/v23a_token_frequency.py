import pandas as pd
import re
import unicodedata
import gc
import heapq
import time
from collections import Counter, defaultdict

# =========================================================
# SETTINGS
# =========================================================
CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
TOKEN_MAX_FREQ = 5000  # WIDENED FROM 100 TO ISOLATE SIGNAL
MAX_CANDIDATES = 100   # KEPT AT 100 TO MATCH V22 BUDGET

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

if __name__ == "__main__":
    t0 = time.time()
    
    # ---------------------------------------------------------
    # LOAD S1 & GROUND TRUTH
    # ---------------------------------------------------------
    print("Loading Validation S1 & Ground Truth...")
    val_ids = set(pd.read_csv("dataset/train/validation_s1_ids.tsv", sep="\t")["source1_entity_id"].astype(str))

    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1 = s1[s1["entity_id"].astype(str).isin(val_ids)]

    s1_map = {}
    s1_ids_list = []
    for i, row in enumerate(s1.itertuples(index=False)):
        old_name, old_addr = normalize_text(row.business_name), normalize_text(row.business_address)
        core_name, strong_addr = strong_name_core(row.business_name), strong_address(row.business_address)
        s1_map[str(row.entity_id)] = {
            "index": i, "country": str(row.country), "old_name": old_name, 
            "old_name_tokens": tokens(old_name), "old_address_tokens": tokens(old_addr),
            "core_name": core_name, "strong_address": strong_addr,
            "name_grams": char3grams(core_name), "address_grams": char3grams(strong_addr),
        }
        s1_ids_list.append(str(row.entity_id))
    del s1; gc.collect()

    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt["source1_entity_id"].astype(str).isin(s1_map.keys())]

    true_pairs = set()
    for row in gt.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not str(row.matched_entity_ids).strip(): continue
        sid = str(row.source1_entity_id)
        for tid in str(row.matched_entity_ids).split(","):
            true_pairs.add((sid, str(tid)))
    del gt; gc.collect()

    # ---------------------------------------------------------
    # PASS 1: TOKEN & GRAM COUNTS
    # ---------------------------------------------------------
    print("\nPASS 1: Counting Tokens & Grams...")
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

    count_stats("dataset/train/train_source2.tsv")
    count_stats("dataset/train/train_source3.tsv")

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
        n_cands.sort()
        for freq, g in n_cands[:3]: 
            name_gram_lookup[(c, g)].append(idx)
            
        a_cands = [(address_gram_counts[c].get(g, 0), g) for g in row["address_grams"] if 0 < address_gram_counts[c].get(g, 0) <= GRAM_MAX_FREQ]
        a_cands.sort()
        for freq, g in a_cands[:3]: 
            address_gram_lookup[(c, g)].append(idx)

    # ---------------------------------------------------------
    # PASS 2: BLOCKING CANDIDATE GENERATION
    # ---------------------------------------------------------
    print("\nPASS 2: Generating Candidates & Measuring Recall...")
    heaps = [[] for _ in range(len(s1_map))]

    def process_blocking(path):
        for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], chunksize=CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                c, tid = str(row.country), str(row.entity_id)
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
                    if len(heap) < MAX_CANDIDATES:
                        heapq.heappush(heap, (score, tid))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, tid))
            gc.collect()

    process_blocking("dataset/train/train_source2.tsv")
    process_blocking("dataset/train/train_source3.tsv")

    # ---------------------------------------------------------
    # CALCULATE ORACLE RECALL
    # ---------------------------------------------------------
    retrieved_pairs = set()
    for idx, heap in enumerate(heaps):
        sid = str(s1_ids_list[idx])
        for _, tid in heap:
            retrieved_pairs.add((sid, str(tid)))

    overlap = true_pairs & retrieved_pairs
    recall = len(overlap) / len(true_pairs) if true_pairs else 0

    print("\n" + "="*50)
    print(f"V23A ORACLE RECALL DIAGNOSTIC (Token Freq={TOKEN_MAX_FREQ}, Top-{MAX_CANDIDATES})")
    print("="*50)
    print(f"Total True Validation Pairs: {len(true_pairs):,}")
    print(f"True Pairs inside Top-{MAX_CANDIDATES}:   {len(overlap):,}")
    print(f"True Pairs Missed Entirely:  {len(true_pairs) - len(overlap):,}")
    print(f"Oracle Candidate Recall:     {recall:.4%}")
    print(f"Runtime:                     {(time.time() - t0)/60:.2f} minutes")
    print("="*50)