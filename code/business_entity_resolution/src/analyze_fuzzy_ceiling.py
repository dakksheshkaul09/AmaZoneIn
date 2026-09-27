import pandas as pd
import re
import unicodedata
import gc

from collections import Counter, defaultdict
from rapidfuzz import fuzz


# =========================================================
# SETTINGS
# =========================================================

ADDRESS_FREQ_LIMIT = 100
NAME_FREQ_LIMIT = 100

CHUNK_SIZE = 200_000


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
        " " if unicodedata.category(c).startswith("P")
        else c
        for c in x
    )

    return " ".join(x.split())


def tokens(x):

    if not x:
        return set()

    return set(
        re.findall(r"\w+", x)
    )


# =========================================================
# 2. STRONG NAME NORMALIZATION
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
    "lp": "lp"
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
    "lp"
}


def strong_name(x):

    x = normalize_text(x)

    if not x:
        return ""

    result = []

    for token in x.split():

        result.append(
            LEGAL_SUFFIX_MAP.get(
                token,
                token
            )
        )

    return " ".join(result)


def strong_name_core(x):

    x = strong_name(x)

    if not x:
        return ""

    token_set = set(
        x.split()
    )

    token_set -= LEGAL_SUFFIXES

    return " ".join(
        sorted(token_set)
    )


# =========================================================
# 3. ADDRESS NORMALIZATION
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
    "number": "number"
}


def strong_address_string(x):

    x = normalize_text(x)

    if not x:
        return ""

    result = []

    for token in x.split():

        result.append(
            ADDRESS_MAP.get(
                token,
                token
            )
        )

    return " ".join(
        sorted(set(result))
    )


# =========================================================
# 4. LOAD VALIDATION IDS
# =========================================================

print("Loading validation IDs...")

val_ids_df = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids_df["source1_entity_id"]
)

del val_ids_df


# =========================================================
# 5. LOAD VALIDATION SOURCE 1
# =========================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()


s1_map = {}

for row in s1.itertuples(index=False):

    old_name = normalize_text(
        row.business_name
    )

    old_address = normalize_text(
        row.business_address
    )

    s1_map[row.entity_id] = {

        "country": row.country,

        "old_name": old_name,

        "old_name_tokens":
            tokens(old_name),

        "old_address_tokens":
            tokens(old_address),

        "strong_name":
            strong_name(
                row.business_name
            ),

        "strong_name_core":
            strong_name_core(
                row.business_name
            ),

        "strong_address":
            strong_address_string(
                row.business_address
            )
    }


print(
    "Validation S1:",
    len(s1_map)
)

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
        "matched_entity_ids"
    ]
)

gt = gt[
    gt["source1_entity_id"].isin(
        s1_map.keys()
    )
].copy()


gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
].copy()


# =========================================================
# 7. TRUE TARGET -> SOURCE 1
# =========================================================

print("Building true-pair lookup...")

pairs = gt.assign(
    matched_entity_ids=
    gt["matched_entity_ids"].str.split(",")
).explode(
    "matched_entity_ids"
)

pairs = pairs.rename(
    columns={
        "matched_entity_ids":
            "target_id"
    }
)

target_to_s1 = dict(
    zip(
        pairs["target_id"],
        pairs["source1_entity_id"]
    )
)


print(
    "True validation pairs:",
    len(target_to_s1)
)

del gt
del pairs
gc.collect()


# =========================================================
# 8. PASS 1 — TOKEN FREQUENCIES
# =========================================================

print("\n" + "=" * 70)
print("PASS 1 — COUNTING BLOCKER TOKENS")
print("=" * 70)

name_counts = defaultdict(Counter)
address_counts = defaultdict(Counter)


def count_tokens(path):

    print("\nScanning:", path)

    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):

            country = row.country

            old_name = normalize_text(
                row.business_name
            )

            for token in tokens(old_name):

                name_counts[country][token] += 1

            old_address = normalize_text(
                row.business_address
            )

            for token in tokens(old_address):

                address_counts[country][token] += 1

        processed += len(chunk)

        if processed % 1_000_000 == 0:
            print(
                f"Processed {processed:,}"
            )

        del chunk

    gc.collect()


count_tokens("train_source2.tsv")
count_tokens("train_source3.tsv")


# =========================================================
# 9. STATISTICS
# =========================================================

total_pairs = len(target_to_s1)

current_blocked = 0

missed_pairs = 0

name_ratio_values = []
name_token_values = []
core_ratio_values = []
address_ratio_values = []
address_token_values = []


# =========================================================
# 10. PASS 2 — FUZZY ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("PASS 2 — FUZZY ANALYSIS OF CURRENT V2 MISSES")
print("=" * 70)


def analyze_file(path):

    global current_blocked
    global missed_pairs

    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):

            target_id = row.entity_id

            if target_id not in target_to_s1:
                continue

            s1_id = target_to_s1[
                target_id
            ]

            s1_row = s1_map[s1_id]

            country = row.country

            old_name = normalize_text(
                row.business_name
            )

            old_address = normalize_text(
                row.business_address
            )

            old_name_tokens = tokens(
                old_name
            )

            old_address_tokens = tokens(
                old_address
            )


            # =================================================
            # CURRENT V2 BLOCKER
            # =================================================

            blocked = False


            # Block A: exact normalized name
            if (
                country ==
                s1_row["country"]
                and
                old_name ==
                s1_row["old_name"]
            ):

                blocked = True


            # Block B: rare address token
            if not blocked:

                if country == s1_row["country"]:

                    shared_address = (
                        s1_row["old_address_tokens"]
                        &
                        old_address_tokens
                    )

                    counter = address_counts[
                        country
                    ]

                    for token in shared_address:

                        if (
                            counter.get(
                                token,
                                0
                            )
                            <=
                            ADDRESS_FREQ_LIMIT
                        ):

                            blocked = True
                            break


            # Block C: rare name token
            if not blocked:

                if country == s1_row["country"]:

                    shared_name = (
                        s1_row["old_name_tokens"]
                        &
                        old_name_tokens
                    )

                    counter = name_counts[
                        country
                    ]

                    for token in shared_name:

                        if (
                            counter.get(
                                token,
                                0
                            )
                            <=
                            NAME_FREQ_LIMIT
                        ):

                            blocked = True
                            break


            if blocked:

                current_blocked += 1
                continue


            # =================================================
            # THIS IS ONE OF THE 87K MISSED TRUE PAIRS
            # =================================================

            missed_pairs += 1


            # =================================================
            # FUZZY NAME
            # =================================================

            name_ratio = fuzz.ratio(
                s1_row["old_name"],
                old_name
            )

            name_token_ratio = fuzz.token_set_ratio(
                s1_row["old_name"],
                old_name
            )


            # =================================================
            # FUZZY STRONG CORE NAME
            # =================================================

            new_core = strong_name_core(
                row.business_name
            )

            core_ratio = fuzz.ratio(
                s1_row["strong_name_core"],
                new_core
            )


            # =================================================
            # FUZZY ADDRESS
            # =================================================

            new_address = strong_address_string(
                row.business_address
            )

            address_ratio = fuzz.ratio(
                s1_row["strong_address"],
                new_address
            )

            address_token_ratio = fuzz.token_set_ratio(
                s1_row["strong_address"],
                new_address
            )


            name_ratio_values.append(
                name_ratio
            )

            name_token_values.append(
                name_token_ratio
            )

            core_ratio_values.append(
                core_ratio
            )

            address_ratio_values.append(
                address_ratio
            )

            address_token_values.append(
                address_token_ratio
            )


        processed += len(chunk)

        if processed % 1_000_000 == 0:

            print(
                f"Processed {processed:,}"
            )

        del chunk

    gc.collect()


analyze_file(
    "train_source2.tsv"
)

analyze_file(
    "train_source3.tsv"
)


# =========================================================
# 11. RESULTS
# =========================================================

print("\n" + "=" * 80)
print("FUZZY CEILING RESULTS")
print("=" * 80)

print(
    "Total true validation pairs:",
    total_pairs
)

print(
    "Current V2 blocked:",
    current_blocked
)

print(
    "Current V2 misses:",
    missed_pairs
)


def print_thresholds(
    values,
    label
):

    print("\n" + label)

    for threshold in [
        70,
        75,
        80,
        85,
        90,
        92,
        95,
        97,
        99
    ]:

        count = sum(
            x >= threshold
            for x in values
        )

        print(
            f">= {threshold}: "
            f"{count:,} "
            f"({count / missed_pairs:.2%})"
        )


print_thresholds(
    name_ratio_values,
    "NAME FUZZ.RATIO"
)

print_thresholds(
    name_token_values,
    "NAME FUZZ.TOKEN_SET_RATIO"
)

print_thresholds(
    core_ratio_values,
    "STRONG CORE NAME FUZZ.RATIO"
)

print_thresholds(
    address_ratio_values,
    "ADDRESS FUZZ.RATIO"
)

print_thresholds(
    address_token_values,
    "ADDRESS FUZZ.TOKEN_SET_RATIO"
)


# =========================================================
# 12. COMBINATIONS
# =========================================================

print("\n" + "=" * 80)
print("COMBINED FUZZY SIGNALS")
print("=" * 80)


for threshold in [
    80,
    85,
    90,
    92,
    95
]:

    count = 0

    for i in range(
        len(name_ratio_values)
    ):

        if (
            name_ratio_values[i] >= threshold
            or
            name_token_values[i] >= threshold
            or
            core_ratio_values[i] >= threshold
            or
            address_ratio_values[i] >= threshold
            or
            address_token_values[i] >= threshold
        ):

            count += 1

    print(
        f"ANY signal >= {threshold}: "
        f"{count:,} "
        f"({count / missed_pairs:.2%} of current misses)"
    )


# =========================================================
# 13. POTENTIAL TOTAL RECALL
# =========================================================

print("\n" + "=" * 80)
print("POTENTIAL TOTAL RECALL")
print("=" * 80)


for threshold in [
    80,
    85,
    90,
    92,
    95
]:

    fuzzy_recovered = 0

    for i in range(
        len(name_ratio_values)
    ):

        if (
            name_ratio_values[i] >= threshold
            or
            name_token_values[i] >= threshold
            or
            core_ratio_values[i] >= threshold
            or
            address_ratio_values[i] >= threshold
            or
            address_token_values[i] >= threshold
        ):

            fuzzy_recovered += 1


    total_recovered = (
        current_blocked
        +
        fuzzy_recovered
    )

    recall = (
        total_recovered /
        total_pairs
    )

    print(
        f"Threshold {threshold}: "
        f"{total_recovered:,} / "
        f"{total_pairs:,} = "
        f"{recall:.2%}"
    )


print("\nDONE.")