import pandas as pd
import re
import unicodedata
import gc
import heapq

from collections import Counter, defaultdict


# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 200_000
GRAM_MAX_FREQ = 5000
MAX_CANDIDATES = 200


# =========================================================
# 1. BASIC NORMALIZATION
# =========================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).casefold()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )
    x = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in x
    )
    return " ".join(x.split())


def tokens(x):
    if not x:
        return set()
    return set(re.findall(r"\w+", x))


# =========================================================
# 2. STRONG NAME
# =========================================================

LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "private": "private",
    "ltd": "limited",
    "limited": "limited",
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "lp": "lp",
}

LEGAL_SUFFIXES = {
    "private",
    "limited",
    "corporation",
    "incorporated",
    "company",
    "llc",
    "llp",
    "plc",
    "lp",
}


def strong_name(x):
    x = normalize_text(x)
    if not x:
        return ""

    return " ".join(
        LEGAL_SUFFIX_MAP.get(token, token)
        for token in x.split()
    )


def strong_name_core(x):
    x = strong_name(x)
    if not x:
        return ""

    token_set = set(x.split())
    token_set -= LEGAL_SUFFIXES
    return " ".join(sorted(token_set))


# =========================================================
# 3. STRONG ADDRESS
# =========================================================

ADDRESS_MAP = {
    "rd": "road",
    "road": "road",
    "st": "street",
    "street": "street",
    "ave": "avenue",
    "av": "avenue",
    "avenue": "avenue",
    "dr": "drive",
    "drive": "drive",
    "ln": "lane",
    "lane": "lane",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "hwy": "highway",
    "highway": "highway",
    "pkwy": "parkway",
    "parkway": "parkway",
    "ct": "court",
    "court": "court",
    "cir": "circle",
    "circle": "circle",
    "pl": "place",
    "place": "place",
    "ter": "terrace",
    "terrace": "terrace",
    "trl": "trail",
    "trail": "trail",
    "apt": "apartment",
    "appt": "apartment",
    "apartment": "apartment",
    "fl": "floor",
    "floor": "floor",
    "rm": "room",
    "room": "room",
    "ste": "suite",
    "suite": "suite",
    "bldg": "building",
    "building": "building",
    "no": "number",
    "num": "number",
    "number": "number",
}


def strong_address(x):
    x = normalize_text(x)
    if not x:
        return ""

    return " ".join(
        sorted({
            ADDRESS_MAP.get(token, token)
            for token in x.split()
        })
    )


# =========================================================
# 4. CHARACTER 3-GRAMS
# =========================================================

def char3grams(x):
    if not x:
        return set()

    compact = "".join(
        c for c in x
        if c.isalnum()
    )

    if len(compact) < 3:
        return set()

    return {
        compact[i:i + 3]
        for i in range(len(compact) - 2)
    }


# =========================================================
# 5. LOAD VALIDATION IDS + S1
# =========================================================

print("Loading validation IDs...")

val_ids_df = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t",
    usecols=["source1_entity_id"],
)

val_ids = set(val_ids_df["source1_entity_id"])

del val_ids_df


print("Loading Source 1...")

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

s1 = s1[s1["entity_id"].isin(val_ids)]

s1_map = {}

for row in s1.itertuples(index=False):
    old_name = normalize_text(row.business_name)
    old_address = normalize_text(row.business_address)
    core_name = strong_name_core(row.business_name)
    strong_addr = strong_address(row.business_address)

    s1_map[row.entity_id] = {
        "country": row.country,
        "old_name": old_name,
        "old_name_tokens": tokens(old_name),
        "old_address_tokens": tokens(old_address),
        "core_name": core_name,
        "strong_address": strong_addr,
        "name_grams": char3grams(core_name),
        "address_grams": char3grams(strong_addr),
    }

print("Validation S1:", len(s1_map))

del s1

gc.collect()


# =========================================================
# 6. GROUND TRUTH
# =========================================================

print("Loading ground truth...")

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

gt = gt[gt["source1_entity_id"].isin(s1_map.keys())]
gt = gt[gt["matched_entity_ids"].fillna("") != ""]

pairs = (
    gt.assign(
        matched_entity_ids=gt["matched_entity_ids"].str.split(",")
    )
    .explode("matched_entity_ids")
    .rename(columns={"matched_entity_ids": "target_id"})
)

target_to_s1 = dict(
    zip(
        pairs["target_id"],
        pairs["source1_entity_id"],
    )
)

print("True validation pairs:", len(target_to_s1))

del gt

del pairs

gc.collect()


# =========================================================
# 7. ONE PASS: COUNT ALL OLD TOKENS + ALL CHAR3GRAMS
# =========================================================

print("\n" + "=" * 70)
print("PASS 1 — COUNTING TOKENS + CHARACTER 3-GRAMS")
print("=" * 70)

name_counts = defaultdict(Counter)
address_counts = defaultdict(Counter)
name_gram_counts = defaultdict(Counter)
address_gram_counts = defaultdict(Counter)


def count_everything(path):
    print("\nScanning:", path)
    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNK_SIZE,
    ):
        for row in chunk.itertuples(index=False):
            country = row.country

            old_name = normalize_text(row.business_name)
            for token in tokens(old_name):
                name_counts[country][token] += 1

            old_address = normalize_text(row.business_address)
            for token in tokens(old_address):
                address_counts[country][token] += 1

            core_name = strong_name_core(row.business_name)
            for gram in char3grams(core_name):
                name_gram_counts[country][gram] += 1

            strong_addr = strong_address(row.business_address)
            for gram in char3grams(strong_addr):
                address_gram_counts[country][gram] += 1

        processed += len(chunk)

        if processed % 1_000_000 == 0:
            print(f"Processed {processed:,}")

        del chunk

    gc.collect()


count_everything("train_source2.tsv")
count_everything("train_source3.tsv")


# =========================================================
# 8. BUILD S1 REVERSE LOOKUPS
# =========================================================

print("\n" + "=" * 70)
print("BUILDING S1 REVERSE LOOKUPS")
print("=" * 70)

exact_name_lookup = defaultdict(list)
core_name_lookup = defaultdict(list)
strong_address_lookup = defaultdict(list)
rare_name_token_lookup = defaultdict(list)
rare_address_token_lookup = defaultdict(list)
name_gram_lookup = defaultdict(list)
address_gram_lookup = defaultdict(list)


for s1_id, row in s1_map.items():
    country = row["country"]

    if row["old_name"]:
        exact_name_lookup[(country, row["old_name"])].append(s1_id)

    if row["core_name"]:
        core_name_lookup[(country, row["core_name"])].append(s1_id)

    if row["strong_address"]:
        strong_address_lookup[(country, row["strong_address"])].append(s1_id)

    counter = name_counts[country]
    for token in row["old_name_tokens"]:
        if 0 < counter.get(token, 0) <= 100:
            rare_name_token_lookup[(country, token)].append(s1_id)

    counter = address_counts[country]
    for token in row["old_address_tokens"]:
        if 0 < counter.get(token, 0) <= 100:
            rare_address_token_lookup[(country, token)].append(s1_id)

    # ONLY TOP-1 rare name 3-gram
    candidates = []
    counter = name_gram_counts[country]

    for gram in row["name_grams"]:
        freq = counter.get(gram, 0)
        if 0 < freq <= GRAM_MAX_FREQ:
            candidates.append((freq, gram))

    if candidates:
        candidates.sort()
        name_gram_lookup[
            (country, candidates[0][1])
        ].append(s1_id)

    # ONLY TOP-1 rare address 3-gram
    candidates = []
    counter = address_gram_counts[country]

    for gram in row["address_grams"]:
        freq = counter.get(gram, 0)
        if 0 < freq <= GRAM_MAX_FREQ:
            candidates.append((freq, gram))

    if candidates:
        candidates.sort()
        address_gram_lookup[
            (country, candidates[0][1])
        ].append(s1_id)

print("Exact-name keys:", len(exact_name_lookup))
print("Core-name keys:", len(core_name_lookup))
print("Address keys:", len(strong_address_lookup))
print("Name gram keys:", len(name_gram_lookup))
print("Address gram keys:", len(address_gram_lookup))


# =========================================================
# 9. TRUE TARGETS BY S1
# =========================================================

true_targets_by_s1 = defaultdict(set)

for target_id, s1_id in target_to_s1.items():
    true_targets_by_s1[s1_id].add(target_id)


# =========================================================
# 10. ONLY ONE TOP-200 HEAP PER S1
# =========================================================

# Important optimization:
# We DO NOT keep separate TOP-50 / TOP-100 / TOP-200 structures.
# We keep TOP-200 only and derive the smaller budgets afterward.

heaps = defaultdict(list)


total_targets_seen = 0


# =========================================================
# 11. ADD CANDIDATE
# =========================================================

def add_candidate(s1_id, target_id, score):
    heap = heaps[s1_id]

    item = (score, target_id)

    if len(heap) < MAX_CANDIDATES:
        heapq.heappush(heap, item)
        return

    if score > heap[0][0]:
        heapq.heapreplace(heap, item)


# =========================================================
# 12. PROCESS TARGET FILE
# =========================================================

def process_targets(path):
    global total_targets_seen

    print("\nScanning:", path)
    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNK_SIZE,
    ):
        for row in chunk.itertuples(index=False):
            total_targets_seen += 1

            target_id = row.entity_id
            country = row.country

            old_name = normalize_text(row.business_name)
            old_address = normalize_text(row.business_address)
            core_name = strong_name_core(row.business_name)
            strong_addr = strong_address(row.business_address)

            old_name_tokens = tokens(old_name)
            old_address_tokens = tokens(old_address)
            name_grams = char3grams(core_name)
            address_grams = char3grams(strong_addr)

            # One temporary score dictionary for THIS target only.
            # This avoids duplicate target entries in the heap.
            scores = {}

            def touch(sid, amount):
                scores[sid] = scores.get(sid, 0.0) + amount

            # -------------------------------------------------
            # A. Exact normalized name
            # -------------------------------------------------
            for sid in exact_name_lookup.get(
                (country, old_name), []
            ):
                touch(sid, 100.0)

            # -------------------------------------------------
            # B. Exact strong core name
            # -------------------------------------------------
            for sid in core_name_lookup.get(
                (country, core_name), []
            ):
                touch(sid, 80.0)

            # -------------------------------------------------
            # C. Exact strong address
            # -------------------------------------------------
            for sid in strong_address_lookup.get(
                (country, strong_addr), []
            ):
                touch(sid, 60.0)

            # -------------------------------------------------
            # D. Rare name token <=100
            # -------------------------------------------------
            counter = name_counts[country]

            for token in old_name_tokens:
                freq = counter.get(token, 0)

                if 0 < freq <= 100:
                    weight = 15.0 / (1.0 + freq ** 0.5)

                    for sid in rare_name_token_lookup.get(
                        (country, token), []
                    ):
                        touch(sid, weight)

            # -------------------------------------------------
            # E. Rare address token <=100
            # -------------------------------------------------
            counter = address_counts[country]

            for token in old_address_tokens:
                freq = counter.get(token, 0)

                if 0 < freq <= 100:
                    weight = 20.0 / (1.0 + freq ** 0.5)

                    for sid in rare_address_token_lookup.get(
                        (country, token), []
                    ):
                        touch(sid, weight)

            # -------------------------------------------------
            # F. TOP-1 RARE NAME CHARACTER 3-GRAM
            # -------------------------------------------------
            counter = name_gram_counts[country]

            for gram in name_grams:
                freq = counter.get(gram, 0)

                if 0 < freq <= GRAM_MAX_FREQ:
                    weight = 40.0 / (1.0 + freq ** 0.5)

                    for sid in name_gram_lookup.get(
                        (country, gram), []
                    ):
                        touch(sid, weight)

            # -------------------------------------------------
            # G. TOP-1 RARE ADDRESS CHARACTER 3-GRAM
            # -------------------------------------------------
            counter = address_gram_counts[country]

            for gram in address_grams:
                freq = counter.get(gram, 0)

                if 0 < freq <= GRAM_MAX_FREQ:
                    weight = 50.0 / (1.0 + freq ** 0.5)

                    for sid in address_gram_lookup.get(
                        (country, gram), []
                    ):
                        touch(sid, weight)

            # -------------------------------------------------
            # Push each candidate exactly once for this target.
            # -------------------------------------------------
            for sid, score in scores.items():
                add_candidate(
                    sid,
                    target_id,
                    score,
                )

        processed += len(chunk)

        if processed % 1_000_000 == 0:
            print(f"Processed {processed:,}")

        del chunk

    gc.collect()


process_targets("train_source2.tsv")
process_targets("train_source3.tsv")


# =========================================================
# 13. RESULTS
# =========================================================

print("\n" + "=" * 80)
print("V3 FAST CANDIDATE BUDGET RESULTS")
print("=" * 80)

print("Total targets scanned:", f"{total_targets_seen:,}")
print("True validation pairs:", f"{len(target_to_s1):,}")


for budget in [50, 100, 200]:

    total_candidates = 0
    s1_with_candidates = 0
    recovered_pairs = 0
    candidate_counts = []

    for s1_id in s1_map:
        heap = heaps.get(s1_id, [])

        # heap contains top 200. Take its best `budget` entries.
        if len(heap) > budget:
            selected = heapq.nlargest(
                budget,
                heap,
                key=lambda x: x[0],
            )
        else:
            selected = list(heap)

        count = len(selected)
        candidate_counts.append(count)
        total_candidates += count

        if count > 0:
            s1_with_candidates += 1

        selected_ids = {
            target_id
            for _, target_id in selected
        }

        true_ids = true_targets_by_s1.get(
            s1_id,
            set(),
        )

        recovered_pairs += len(
            selected_ids & true_ids
        )

    candidate_counts.sort()

    def percentile(values, p):
        if not values:
            return 0

        index = int(len(values) * p)

        if index >= len(values):
            index = len(values) - 1

        return values[index]

    avg_candidates = (
        total_candidates /
        len(s1_map)
    )

    candidate_recall = (
        recovered_pairs /
        len(target_to_s1)
    )

    print("\n" + "-" * 70)
    print(f"TOP {budget} CANDIDATES PER S1")
    print("Total candidates:", f"{total_candidates:,}")
    print("S1 with >=1 candidate:", f"{s1_with_candidates:,}")
    print("Average candidates/S1:", f"{avg_candidates:.2f}")
    print("P95 candidates/S1:", percentile(candidate_counts, 0.95))
    print("P99 candidates/S1:", percentile(candidate_counts, 0.99))
    print("Recovered true pairs:", f"{recovered_pairs:,}")
    print("Candidate recall:", f"{candidate_recall:.4%}")


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
