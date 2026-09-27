import pandas as pd
import re
import unicodedata
import gc

from collections import Counter, defaultdict


# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 200_000

# Character n-gram size
N = 3

# Rare n-gram thresholds to test
RARE_LIMITS = [
    10,
    50,
    100,
    500,
    1000,
    5000
]


# =========================================================
# 1. BASIC NORMALIZATION
# =========================================================

def normalize_text(x):

    if pd.isna(x):
        return ""

    x = str(x).casefold()

    x = unicodedata.normalize(
        "NFKD",
        x
    )

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
        re.findall(
            r"\w+",
            x
        )
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
# 3. STRONG ADDRESS NORMALIZATION
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
# 4. CHARACTER N-GRAMS
# =========================================================

def char_ngrams(x):

    if not x:
        return set()

    # Remove spaces so small word-boundary changes
    # don't destroy the n-grams.
    compact = "".join(
        c for c in x
        if c.isalnum()
    )

    if len(compact) < N:
        return set()

    return {
        compact[i:i + N]
        for i in range(
            len(compact) - N + 1
        )
    }


# =========================================================
# 5. LOAD VALIDATION IDS
# =========================================================

print("Loading validation IDs...")

val_ids_df = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids_df[
        "source1_entity_id"
    ]
)

del val_ids_df


# =========================================================
# 6. LOAD VALIDATION SOURCE 1
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

    s1_map[
        row.entity_id
    ] = {

        "country":
            row.country,

        "old_name":
            old_name,

        "old_name_tokens":
            tokens(old_name),

        "old_address_tokens":
            tokens(old_address),

        "name_ngrams":
            char_ngrams(
                strong_name_core(
                    row.business_name
                )
            ),

        "address_ngrams":
            char_ngrams(
                strong_address_string(
                    row.business_address
                )
            )
    }


print(
    "Validation S1:",
    len(s1_map)
)

del s1
gc.collect()


# =========================================================
# 7. LOAD GROUND TRUTH
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
    gt[
        "source1_entity_id"
    ].isin(
        s1_map.keys()
    )
].copy()


gt = gt[
    gt[
        "matched_entity_ids"
    ].fillna("") != ""
].copy()


# =========================================================
# 8. TRUE TARGET -> SOURCE 1
# =========================================================

print("Building true-pair lookup...")

pairs = gt.assign(
    matched_entity_ids=
    gt[
        "matched_entity_ids"
    ].str.split(",")
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
        pairs[
            "target_id"
        ],
        pairs[
            "source1_entity_id"
        ]
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
# 9. PASS 1
#    COUNT TOKEN FREQUENCIES
#    TO REPRODUCE CURRENT V2
# =========================================================

print("\n" + "=" * 70)
print("PASS 1A — OLD TOKEN FREQUENCIES")
print("=" * 70)


name_counts = defaultdict(Counter)
address_counts = defaultdict(Counter)


def count_old_tokens(path):

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

        for row in chunk.itertuples(
            index=False
        ):

            country = row.country

            old_name = normalize_text(
                row.business_name
            )

            for token in tokens(
                old_name
            ):

                name_counts[
                    country
                ][
                    token
                ] += 1

            old_address = normalize_text(
                row.business_address
            )

            for token in tokens(
                old_address
            ):

                address_counts[
                    country
                ][
                    token
                ] += 1

        processed += len(chunk)

        if processed % 1_000_000 == 0:

            print(
                f"Processed {processed:,}"
            )

        del chunk

    gc.collect()


count_old_tokens(
    "train_source2.tsv"
)

count_old_tokens(
    "train_source3.tsv"
)


# =========================================================
# 10. PASS 1B
#     CHARACTER 3-GRAM FREQUENCIES
# =========================================================

print("\n" + "=" * 70)
print("PASS 1B — CHARACTER 3-GRAM FREQUENCIES")
print("=" * 70)

name_ngram_counts = defaultdict(Counter)
address_ngram_counts = defaultdict(Counter)


def count_char_ngrams(path):

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

        for row in chunk.itertuples(
            index=False
        ):

            country = row.country


            # -----------------------------
            # Strong core name
            # -----------------------------

            name = strong_name_core(
                row.business_name
            )

            for gram in char_ngrams(
                name
            ):

                name_ngram_counts[
                    country
                ][
                    gram
                ] += 1


            # -----------------------------
            # Strong address
            # -----------------------------

            address = strong_address_string(
                row.business_address
            )

            for gram in char_ngrams(
                address
            ):

                address_ngram_counts[
                    country
                ][
                    gram
                ] += 1

        processed += len(chunk)

        if processed % 1_000_000 == 0:

            print(
                f"Processed {processed:,}"
            )

        del chunk

    gc.collect()


count_char_ngrams(
    "train_source2.tsv"
)

count_char_ngrams(
    "train_source3.tsv"
)


print("\nCharacter 3-gram counting complete.")


# =========================================================
# 11. STATISTICS
# =========================================================

total_pairs = len(
    target_to_s1
)

current_blocked = 0
current_misses = 0


# ---------------------------------------------------------
# For each threshold we store:
# name recovered
# address recovered
# either recovered
# ---------------------------------------------------------

name_recovery = {
    limit: 0
    for limit in RARE_LIMITS
}

address_recovery = {
    limit: 0
    for limit in RARE_LIMITS
}

combined_recovery = {
    limit: 0
    for limit in RARE_LIMITS
}


# Number of shared rare grams required
# We test 1, 2, 3
name_multi_recovery = {
    limit: {
        1: 0,
        2: 0,
        3: 0
    }
    for limit in RARE_LIMITS
}

address_multi_recovery = {
    limit: {
        1: 0,
        2: 0,
        3: 0
    }
    for limit in RARE_LIMITS
}


# =========================================================
# 12. CANDIDATE SIZE ESTIMATE
#     OVER ALL VALIDATION S1
# =========================================================

print("\n" + "=" * 70)
print("ESTIMATING CHARACTER-BLOCK CANDIDATE SIZE")
print("=" * 70)


candidate_sum_name = {
    limit: []
    for limit in RARE_LIMITS
}

candidate_sum_address = {
    limit: []
    for limit in RARE_LIMITS
}


for s1_id, row in s1_map.items():

    country = row["country"]

    name_counter = name_ngram_counts[
        country
    ]

    address_counter = address_ngram_counts[
        country
    ]


    for limit in RARE_LIMITS:

        name_sum = 0

        for gram in row[
            "name_ngrams"
        ]:

            freq = name_counter.get(
                gram,
                0
            )

            if (
                freq > 0
                and
                freq <= limit
            ):

                name_sum += freq


        address_sum = 0

        for gram in row[
            "address_ngrams"
        ]:

            freq = address_counter.get(
                gram,
                0
            )

            if (
                freq > 0
                and
                freq <= limit
            ):

                address_sum += freq


        candidate_sum_name[
            limit
        ].append(
            name_sum
        )

        candidate_sum_address[
            limit
        ].append(
            address_sum
        )


print("\nNOTE:")
print(
    "Candidate sums are an UPPER BOUND"
)
print(
    "because multiple n-grams may retrieve"
)
print(
    "the same target record."
)


for limit in RARE_LIMITS:

    name_values = sorted(
        candidate_sum_name[
            limit
        ]
    )

    address_values = sorted(
        candidate_sum_address[
            limit
        ]
    )


    def percentile(
        values,
        p
    ):

        index = int(
            len(values) * p
        )

        if index >= len(values):
            index = len(values) - 1

        return values[
            index
        ]


    print(
        f"\nRare frequency <= {limit}"
    )

    print(
        "NAME:"
    )

    print(
        "  S1 with >=1 posting:",
        sum(
            x > 0
            for x in name_values
        )
    )

    print(
        "  Avg upper-bound candidates:",
        sum(name_values) /
        len(name_values)
    )

    print(
        "  P95:",
        percentile(
            name_values,
            0.95
        )
    )

    print(
        "  P99:",
        percentile(
            name_values,
            0.99
        )
    )


    print(
        "ADDRESS:"
    )

    print(
        "  S1 with >=1 posting:",
        sum(
            x > 0
            for x in address_values
        )
    )

    print(
        "  Avg upper-bound candidates:",
        sum(address_values) /
        len(address_values)
    )

    print(
        "  P95:",
        percentile(
            address_values,
            0.95
        )
    )

    print(
        "  P99:",
        percentile(
            address_values,
            0.99
        )
    )


# =========================================================
# 13. PASS 2
#     ANALYZE CURRENT V2 MISSES
# =========================================================

print("\n" + "=" * 70)
print("PASS 2 — CHARACTER 3-GRAM CEILING")
print("=" * 70)


def analyze_file(path):

    global current_blocked
    global current_misses

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

        for row in chunk.itertuples(
            index=False
        ):

            target_id = row.entity_id

            if target_id not in target_to_s1:
                continue


            s1_id = target_to_s1[
                target_id
            ]

            s1_row = s1_map[
                s1_id
            ]

            country = row.country


            # =================================================
            # CURRENT V2 BLOCKER
            # =================================================

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


            blocked = False


            # -------------------------------------------------
            # BLOCK A
            # Exact normalized name
            # -------------------------------------------------

            if (
                country ==
                s1_row["country"]

                and

                old_name ==
                s1_row["old_name"]
            ):

                blocked = True


            # -------------------------------------------------
            # BLOCK B
            # Rare address token <=100
            # -------------------------------------------------

            if not blocked:

                if (
                    country ==
                    s1_row["country"]
                ):

                    shared_address = (
                        s1_row[
                            "old_address_tokens"
                        ]
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
                            <= 100
                        ):

                            blocked = True
                            break


            # -------------------------------------------------
            # BLOCK C
            # Rare name token <=100
            # -------------------------------------------------

            if not blocked:

                if (
                    country ==
                    s1_row["country"]
                ):

                    shared_name = (
                        s1_row[
                            "old_name_tokens"
                        ]
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
                            <= 100
                        ):

                            blocked = True
                            break


            if blocked:

                current_blocked += 1
                continue


            # =================================================
            # CURRENT V2 MISS
            # =================================================

            current_misses += 1


            # =================================================
            # TARGET CHARACTER 3-GRAMS
            # =================================================

            target_name = strong_name_core(
                row.business_name
            )

            target_address = strong_address_string(
                row.business_address
            )


            target_name_grams = char_ngrams(
                target_name
            )

            target_address_grams = char_ngrams(
                target_address
            )


            s1_name_grams = s1_row[
                "name_ngrams"
            ]

            s1_address_grams = s1_row[
                "address_ngrams"
            ]


            # =================================================
            # SHARED GRAMS
            # =================================================

            shared_name_grams = (
                s1_name_grams
                &
                target_name_grams
            )

            shared_address_grams = (
                s1_address_grams
                &
                target_address_grams
            )


            name_counter = name_ngram_counts[
                country
            ]

            address_counter = address_ngram_counts[
                country
            ]


            # =================================================
            # TEST EACH RARE THRESHOLD
            # =================================================

            for limit in RARE_LIMITS:


                # ---------------------------------------------
                # Rare NAME grams
                # ---------------------------------------------

                rare_name_shared = [

                    gram

                    for gram
                    in shared_name_grams

                    if (
                        name_counter.get(
                            gram,
                            0
                        )
                        <= limit
                    )
                ]


                if len(
                    rare_name_shared
                ) >= 1:

                    name_recovery[
                        limit
                    ] += 1


                if len(
                    rare_name_shared
                ) >= 1:

                    name_multi_recovery[
                        limit
                    ][1] += 1

                if len(
                    rare_name_shared
                ) >= 2:

                    name_multi_recovery[
                        limit
                    ][2] += 1

                if len(
                    rare_name_shared
                ) >= 3:

                    name_multi_recovery[
                        limit
                    ][3] += 1


                # ---------------------------------------------
                # Rare ADDRESS grams
                # ---------------------------------------------

                rare_address_shared = [

                    gram

                    for gram
                    in shared_address_grams

                    if (
                        address_counter.get(
                            gram,
                            0
                        )
                        <= limit
                    )
                ]


                if len(
                    rare_address_shared
                ) >= 1:

                    address_recovery[
                        limit
                    ] += 1


                if len(
                    rare_address_shared
                ) >= 1:

                    address_multi_recovery[
                        limit
                    ][1] += 1

                if len(
                    rare_address_shared
                ) >= 2:

                    address_multi_recovery[
                        limit
                    ][2] += 1

                if len(
                    rare_address_shared
                ) >= 3:

                    address_multi_recovery[
                        limit
                    ][3] += 1


                # ---------------------------------------------
                # NAME OR ADDRESS
                # ---------------------------------------------

                if (
                    len(rare_name_shared) >= 1
                    or
                    len(rare_address_shared) >= 1
                ):

                    combined_recovery[
                        limit
                    ] += 1


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
# 14. FINAL RESULTS
# =========================================================

print("\n" + "=" * 80)
print("CHARACTER 3-GRAM RESULTS")
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
    current_misses
)

print()


# =========================================================
# 15. NAME RECOVERY
# =========================================================

print("=" * 80)
print("NAME CHARACTER 3-GRAM RECOVERY")
print("=" * 80)

for limit in RARE_LIMITS:

    count = name_recovery[
        limit
    ]

    total = (
        current_blocked
        +
        count
    )

    recall = (
        total /
        total_pairs
    )

    print(
        f"Rare <= {limit:>4}: "
        f"{count:,} misses recovered | "
        f"{count / current_misses:.2%} of misses | "
        f"total recall = {recall:.2%}"
    )


    print(
        f"             "
        f">=2 shared rare grams: "
        f"{name_multi_recovery[limit][2]:,}"
    )

    print(
        f"             "
        f">=3 shared rare grams: "
        f"{name_multi_recovery[limit][3]:,}"
    )


# =========================================================
# 16. ADDRESS RECOVERY
# =========================================================

print("\n" + "=" * 80)
print("ADDRESS CHARACTER 3-GRAM RECOVERY")
print("=" * 80)

for limit in RARE_LIMITS:

    count = address_recovery[
        limit
    ]

    total = (
        current_blocked
        +
        count
    )

    recall = (
        total /
        total_pairs
    )

    print(
        f"Rare <= {limit:>4}: "
        f"{count:,} misses recovered | "
        f"{count / current_misses:.2%} of misses | "
        f"total recall = {recall:.2%}"
    )


    print(
        f"             "
        f">=2 shared rare grams: "
        f"{address_multi_recovery[limit][2]:,}"
    )

    print(
        f"             "
        f">=3 shared rare grams: "
        f"{address_multi_recovery[limit][3]:,}"
    )


# =========================================================
# 17. COMBINED
# =========================================================

print("\n" + "=" * 80)
print("NAME OR ADDRESS CHARACTER 3-GRAM RECOVERY")
print("=" * 80)

for limit in RARE_LIMITS:

    count = combined_recovery[
        limit
    ]

    total = (
        current_blocked
        +
        count
    )

    recall = (
        total /
        total_pairs
    )

    print(
        f"Rare <= {limit:>4}: "
        f"{count:,} recovered | "
        f"{count / current_misses:.2%} of misses | "
        f"total recall = {recall:.2%}"
    )


# =========================================================
# 18. DONE
# =========================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)