
import pandas as pd
import re
import unicodedata
import gc
import heapq
import math

from collections import Counter, defaultdict
from rapidfuzz import fuzz


# =========================================================
# V3 TOP-50 ACTUAL MATCHER TEST
# =========================================================
#
# VALIDATION ONLY.
#
# Candidate generation:
#   - exact normalized name
#   - exact strong core name
#   - exact strong address
#   - rare name token <=100
#   - rare address token <=100
#   - top-1 rare character 3-gram from strong core name
#   - top-1 rare character 3-gram from strong address
#
# Candidate ranking:
#   Same V3 blocking score used in the previous experiment.
#
# Matcher:
#   0.40 * token_set_ratio(name)
# + 0.60 * token_set_ratio(address)
# threshold = 0.80
#
# Only TOP 50 candidates/S1 are retained.
#
# This script calculates the real entity-level macro F0.5
# on the frozen validation split.
# =========================================================


# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 200_000

GRAM_MAX_FREQ = 5000

TOP_K = 50

MATCH_THRESHOLD = 0.80

NAME_WEIGHT = 0.40
ADDRESS_WEIGHT = 0.60

BLOCK_SCALE = 1000
MATCH_SCALE = 1000

MATCH_BITS = 11
MATCH_MASK = (1 << MATCH_BITS) - 1


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
        c
        for c in x
        if not unicodedata.combining(c)
    )

    x = "".join(
        " "
        if unicodedata.category(c).startswith("P")
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
    "number": "number"
}


def strong_address(x):

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
# 4. CHARACTER 3-GRAMS
# =========================================================

def char3grams(x):

    if not x:
        return set()

    compact = "".join(
        c
        for c in x
        if c.isalnum()
    )

    if len(compact) < 3:
        return set()

    return {
        compact[i:i + 3]
        for i in range(
            len(compact) - 2
        )
    }


# =========================================================
# 5. PACK / UNPACK COMPACT HEAP ENTRY
# =========================================================

def pack_entry(
    block_score,
    match_score,
    is_true
):

    block_q = int(
        max(
            0,
            round(
                block_score
                * BLOCK_SCALE
            )
        )
    )

    match_q = int(
        max(
            0,
            min(
                MATCH_MASK,
                round(
                    match_score
                    * MATCH_SCALE
                )
            )
        )
    )

    truth_bit = (
        1
        if is_true
        else 0
    )

    return (
        (
            block_q
            << (MATCH_BITS + 1)
        )
        |
        (
            match_q
            << 1
        )
        |
        truth_bit
    )


def unpack_entry(value):

    truth_bit = value & 1

    match_q = (
        value
        >> 1
    ) & MATCH_MASK

    block_q = (
        value
        >> (MATCH_BITS + 1)
    )

    return (
        block_q / BLOCK_SCALE,
        match_q / MATCH_SCALE,
        bool(truth_bit)
    )


# =========================================================
# 6. ENTITY-LEVEL F0.5
# =========================================================

def entity_f05(
    true_count,
    predicted_count,
    tp
):

    if true_count == 0:

        if predicted_count == 0:
            return 1.0

        return 0.0

    if predicted_count == 0:
        return 0.0

    fp = (
        predicted_count
        - tp
    )

    fn = (
        true_count
        - tp
    )

    precision = (
        tp
        /
        (tp + fp)
    )

    recall = (
        tp
        /
        (tp + fn)
    )

    if (
        precision == 0
        or
        recall == 0
    ):
        return 0.0

    return (
        1.25
        * precision
        * recall
        /
        (
            0.25 * precision
            +
            recall
        )
    )


# =========================================================
# 7. LOAD VALIDATION IDS
# =========================================================

print(
    "Loading validation IDs..."
)

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
# 8. LOAD SOURCE 1
# =========================================================

print(
    "Loading Source 1..."
)

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
    s1[
        "entity_id"
    ].isin(
        val_ids
    )
].copy()


s1_map = {}


for row in s1.itertuples(
    index=False
):

    old_name = normalize_text(
        row.business_name
    )

    old_address = normalize_text(
        row.business_address
    )

    core_name = strong_name_core(
        row.business_name
    )

    strong_addr = strong_address(
        row.business_address
    )

    s1_map[
        row.entity_id
    ] = {

        "country":
            row.country,

        "old_name":
            old_name,

        "old_address":
            old_address,

        "old_name_tokens":
            tokens(old_name),

        "old_address_tokens":
            tokens(old_address),

        "core_name":
            core_name,

        "strong_address":
            strong_addr,

        "name_grams":
            char3grams(core_name),

        "address_grams":
            char3grams(strong_addr)
    }


s1_ids = list(
    s1_map.keys()
)

s1_index = {
    sid: i
    for i, sid
    in enumerate(s1_ids)
}


print(
    "Validation S1:",
    len(s1_map)
)

del s1
gc.collect()


# =========================================================
# 9. LOAD GROUND TRUTH
# =========================================================

print(
    "Loading ground truth..."
)

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

gt[
    "matched_entity_ids"
] = gt[
    "matched_entity_ids"
].fillna("")


true_count_by_s1 = {
    sid: 0
    for sid in s1_map
}

target_to_s1 = {}


for row in gt.itertuples(
    index=False
):

    sid = row.source1_entity_id

    value = row.matched_entity_ids

    if not value:
        continue

    ids = value.split(",")

    true_count_by_s1[
        sid
    ] = len(ids)

    for target_id in ids:

        target_to_s1[
            target_id
        ] = sid


print(
    "True validation pairs:",
    len(target_to_s1)
)

del gt
gc.collect()


# =========================================================
# 10. COUNT TOKENS + CHARACTER 3-GRAMS
# =========================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PASS 1 — COUNTING TOKENS + CHARACTER 3-GRAMS"
)

print(
    "=" * 70
)


name_counts = defaultdict(Counter)
address_counts = defaultdict(Counter)

name_gram_counts = defaultdict(Counter)
address_gram_counts = defaultdict(Counter)


def count_target_statistics(path):

    print(
        "\nScanning:",
        path
    )

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


            name = normalize_text(
                row.business_name
            )

            for token in tokens(name):

                name_counts[
                    country
                ][
                    token
                ] += 1


            address = normalize_text(
                row.business_address
            )

            for token in tokens(address):

                address_counts[
                    country
                ][
                    token
                ] += 1


            core_name = strong_name_core(
                row.business_name
            )

            for gram in char3grams(
                core_name
            ):

                name_gram_counts[
                    country
                ][
                    gram
                ] += 1


            strong_addr = strong_address(
                row.business_address
            )

            for gram in char3grams(
                strong_addr
            ):

                address_gram_counts[
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


count_target_statistics(
    "train_source2.tsv"
)

count_target_statistics(
    "train_source3.tsv"
)


# =========================================================
# 11. BUILD REVERSE LOOKUPS
# =========================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BUILDING V3 REVERSE LOOKUPS"
)

print(
    "=" * 70
)


exact_name_lookup = defaultdict(list)
core_name_lookup = defaultdict(list)
strong_address_lookup = defaultdict(list)

rare_name_token_lookup = defaultdict(list)
rare_address_token_lookup = defaultdict(list)

name_gram_lookup = defaultdict(list)
address_gram_lookup = defaultdict(list)


for sid, row in s1_map.items():

    country = row[
        "country"
    ]


    if row[
        "old_name"
    ]:

        exact_name_lookup[
            (
                country,
                row["old_name"]
            )
        ].append(
            sid
        )


    if row[
        "core_name"
    ]:

        core_name_lookup[
            (
                country,
                row["core_name"]
            )
        ].append(
            sid
        )


    if row[
        "strong_address"
    ]:

        strong_address_lookup[
            (
                country,
                row["strong_address"]
            )
        ].append(
            sid
        )


    counter = name_counts[
        country
    ]

    for token in row[
        "old_name_tokens"
    ]:

        if (
            counter.get(
                token,
                0
            )
            <= 100
        ):

            rare_name_token_lookup[
                (
                    country,
                    token
                )
            ].append(
                sid
            )


    counter = address_counts[
        country
    ]

    for token in row[
        "old_address_tokens"
    ]:

        if (
            counter.get(
                token,
                0
            )
            <= 100
        ):

            rare_address_token_lookup[
                (
                    country,
                    token
                )
            ].append(
                sid
            )


# =========================================================
# 12. TOP-1 RAREST CHARACTER 3-GRAM PER S1
# =========================================================

print(
    "Building TOP-1 rare character grams..."
)


for sid, row in s1_map.items():

    country = row[
        "country"
    ]


    # -----------------------------
    # Name
    # -----------------------------

    candidates = []

    counter = name_gram_counts[
        country
    ]

    for gram in row[
        "name_grams"
    ]:

        freq = counter.get(
            gram,
            0
        )

        if (
            freq > 0
            and
            freq <= GRAM_MAX_FREQ
        ):

            candidates.append(
                (
                    freq,
                    gram
                )
            )

    candidates.sort()

    if candidates:

        top_name_gram = (
            candidates[0][1]
        )

    else:

        top_name_gram = None


    # -----------------------------
    # Address
    # -----------------------------

    candidates = []

    counter = address_gram_counts[
        country
    ]

    for gram in row[
        "address_grams"
    ]:

        freq = counter.get(
            gram,
            0
        )

        if (
            freq > 0
            and
            freq <= GRAM_MAX_FREQ
        ):

            candidates.append(
                (
                    freq,
                    gram
                )
            )

    candidates.sort()

    if candidates:

        top_address_gram = (
            candidates[0][1]
        )

    else:

        top_address_gram = None


    row[
        "top_name_gram"
    ] = top_name_gram

    row[
        "top_address_gram"
    ] = top_address_gram


    if top_name_gram:

        name_gram_lookup[
            (
                country,
                top_name_gram
            )
        ].append(
            sid
        )


    if top_address_gram:

        address_gram_lookup[
            (
                country,
                top_address_gram
            )
        ].append(
            sid
        )


print(
    "Exact-name keys:",
    len(exact_name_lookup)
)

print(
    "Core-name keys:",
    len(core_name_lookup)
)

print(
    "Address keys:",
    len(strong_address_lookup)
)

print(
    "Name gram keys:",
    len(name_gram_lookup)
)

print(
    "Address gram keys:",
    len(address_gram_lookup)
)


# =========================================================
# 13. TOP-50 HEAPS
# =========================================================

heaps = [
    []
    for _ in range(
        len(s1_ids)
    )
]


def update_heap(
    s1_idx,
    block_score,
    matcher_score,
    is_true
):

    heap = heaps[
        s1_idx
    ]

    packed = pack_entry(
        block_score,
        matcher_score,
        is_true
    )

    if len(heap) < TOP_K:

        heapq.heappush(
            heap,
            packed
        )

    elif packed > heap[0]:

        heapq.heapreplace(
            heap,
            packed
        )


# =========================================================
# 14. PROCESS TARGET FILES
# =========================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PASS 2 — V3 CANDIDATE GENERATION + MATCHING"
)

print(
    "=" * 70
)


total_targets_seen = 0
total_candidate_events = 0


def process_targets(path):

    global total_targets_seen
    global total_candidate_events

    print(
        "\nScanning:",
        path
    )

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

            total_targets_seen += 1

            target_id = row.entity_id
            country = row.country


            # Target representations

            old_name = normalize_text(
                row.business_name
            )

            old_address = normalize_text(
                row.business_address
            )

            core_name = strong_name_core(
                row.business_name
            )

            strong_addr = strong_address(
                row.business_address
            )

            old_name_tokens = tokens(
                old_name
            )

            old_address_tokens = tokens(
                old_address
            )

            name_grams = char3grams(
                core_name
            )

            address_grams = char3grams(
                strong_addr
            )


            # Candidate blocking scores for this target

            scores = {}


            def touch(
                sid,
                amount
            ):

                scores[sid] = (
                    scores.get(
                        sid,
                        0.0
                    )
                    +
                    amount
                )


            # A. exact normalized name

            for sid in exact_name_lookup.get(
                (
                    country,
                    old_name
                ),
                []
            ):

                touch(
                    sid,
                    100.0
                )


            # B. exact strong core name

            for sid in core_name_lookup.get(
                (
                    country,
                    core_name
                ),
                []
            ):

                touch(
                    sid,
                    80.0
                )


            # C. exact strong address

            for sid in strong_address_lookup.get(
                (
                    country,
                    strong_addr
                ),
                []
            ):

                touch(
                    sid,
                    60.0
                )


            # D. rare name token <=100

            counter = name_counts[
                country
            ]

            for token in old_name_tokens:

                freq = counter.get(
                    token,
                    0
                )

                if (
                    freq > 0
                    and
                    freq <= 100
                ):

                    weight = (
                        15.0 /
                        (
                            1.0
                            +
                            math.sqrt(freq)
                        )
                    )

                    for sid in rare_name_token_lookup.get(
                        (
                            country,
                            token
                        ),
                        []
                    ):

                        touch(
                            sid,
                            weight
                        )


            # E. rare address token <=100

            counter = address_counts[
                country
            ]

            for token in old_address_tokens:

                freq = counter.get(
                    token,
                    0
                )

                if (
                    freq > 0
                    and
                    freq <= 100
                ):

                    weight = (
                        20.0 /
                        (
                            1.0
                            +
                            math.sqrt(freq)
                        )
                    )

                    for sid in rare_address_token_lookup.get(
                        (
                            country,
                            token
                        ),
                        []
                    ):

                        touch(
                            sid,
                            weight
                        )


            # F. character 3-gram name

            counter = name_gram_counts[
                country
            ]

            for gram in name_grams:

                freq = counter.get(
                    gram,
                    0
                )

                if (
                    freq > 0
                    and
                    freq <= GRAM_MAX_FREQ
                ):

                    weight = (
                        40.0 /
                        (
                            1.0
                            +
                            math.sqrt(freq)
                        )
                    )

                    for sid in name_gram_lookup.get(
                        (
                            country,
                            gram
                        ),
                        []
                    ):

                        touch(
                            sid,
                            weight
                        )


            # G. character 3-gram address

            counter = address_gram_counts[
                country
            ]

            for gram in address_grams:

                freq = counter.get(
                    gram,
                    0
                )

                if (
                    freq > 0
                    and
                    freq <= GRAM_MAX_FREQ
                ):

                    weight = (
                        50.0 /
                        (
                            1.0
                            +
                            math.sqrt(freq)
                        )
                    )

                    for sid in address_gram_lookup.get(
                        (
                            country,
                            gram
                        ),
                        []
                    ):

                        touch(
                            sid,
                            weight
                        )


            # =================================================
            # ACTUAL MATCHER ON ALL RETRIEVED CANDIDATES
            # THEN KEEP TOP-50 BY BLOCKING SCORE
            # =================================================

            for sid, block_score in scores.items():

                total_candidate_events += 1

                s1_row = s1_map[
                    sid
                ]


                name_score = (
                    fuzz.token_set_ratio(
                        s1_row[
                            "old_name"
                        ],
                        old_name
                    )
                    / 100.0
                )


                address_score = (
                    fuzz.token_set_ratio(
                        s1_row[
                            "old_address"
                        ],
                        old_address
                    )
                    / 100.0
                )


                combined_score = (
                    NAME_WEIGHT
                    * name_score
                    +
                    ADDRESS_WEIGHT
                    * address_score
                )


                is_true = (
                    target_to_s1.get(
                        target_id
                    )
                    ==
                    sid
                )


                update_heap(
                    s1_index[sid],
                    block_score,
                    combined_score,
                    is_true
                )


        processed += len(chunk)

        if processed % 1_000_000 == 0:

            print(
                f"Processed {processed:,}"
            )

        del chunk

    gc.collect()


process_targets(
    "train_source2.tsv"
)

process_targets(
    "train_source3.tsv"
)


# =========================================================
# 15. FINAL METRICS
# =========================================================

print(
    "\n"
    + "=" * 80
)

print(
    "V3 TOP-50 ACTUAL MATCHER RESULTS"
)

print(
    "=" * 80
)


total_f05 = 0.0

total_predictions = 0
total_tp = 0
total_fp = 0

s1_with_candidates = 0

candidate_counts = []

singleton_false_positive = 0


for s1_idx, sid in enumerate(
    s1_ids
):

    heap = heaps[
        s1_idx
    ]

    candidate_count = len(
        heap
    )

    candidate_counts.append(
        candidate_count
    )


    if candidate_count > 0:

        s1_with_candidates += 1


    true_count = (
        true_count_by_s1.get(
            sid,
            0
        )
    )


    predicted_count = 0
    tp = 0


    for packed in heap:

        (
            block_score,
            match_score,
            is_true
        ) = unpack_entry(
            packed
        )


        if (
            match_score
            >= MATCH_THRESHOLD
        ):

            predicted_count += 1
            total_predictions += 1


            if is_true:

                tp += 1
                total_tp += 1

            else:

                total_fp += 1


    if (
        true_count == 0
        and
        predicted_count > 0
    ):

        singleton_false_positive += 1


    total_f05 += entity_f05(
        true_count,
        predicted_count,
        tp
    )


candidate_counts.sort()


def percentile(
    values,
    p
):

    if not values:
        return 0

    index = int(
        len(values) * p
    )

    if index >= len(values):

        index = len(values) - 1

    return values[
        index
    ]


macro_f05 = (
    total_f05 /
    len(s1_ids)
)


total_true = len(
    target_to_s1
)


print(
    "Validation S1:",
    len(s1_ids)
)

print(
    "Total true pairs:",
    total_true
)

print(
    "Targets scanned:",
    total_targets_seen
)

print(
    "Candidate events before TOP-50:",
    f"{total_candidate_events:,}"
)

print(
    "S1 with >=1 candidate:",
    s1_with_candidates
)

print(
    "Average candidates/S1:",
    f"{sum(candidate_counts) / len(s1_ids):.2f}"
)

print(
    "P95 candidates/S1:",
    percentile(
        candidate_counts,
        0.95
    )
)

print(
    "P99 candidates/S1:",
    percentile(
        candidate_counts,
        0.99
    )
)

print(
    "Predicted matches:",
    f"{total_predictions:,}"
)

print(
    "True positives:",
    f"{total_tp:,}"
)

print(
    "False positives:",
    f"{total_fp:,}"
)

print(
    "Matcher precision:",
    (
        f"{total_tp / total_predictions:.4%}"
        if total_predictions > 0
        else "0.0000%"
    )
)

print(
    "Pair recall:",
    (
        f"{total_tp / total_true:.4%}"
        if total_true > 0
        else "0.0000%"
    )
)

print(
    "Singletons with false positives:",
    singleton_false_positive
)

print(
    "MACRO F0.5:",
    f"{macro_f05:.8f}"
)

print(
    "Baseline V2 MACRO F0.5:",
    "0.58585938"
)

print(
    "Improvement:",
    f"{macro_f05 - 0.58585938:+.8f}"
)


if macro_f05 > 0.70:

    print(
        "\n"
        ">>> V3 PASSES OUR SUBMISSION #2 "
        "LOCAL THRESHOLD (> 0.70)"
    )

else:

    print(
        "\n"
        ">>> V3 DOES NOT YET PASS "
        "THE >0.70 SUBMISSION #2 THRESHOLD"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "DONE"
)

print(
    "=" * 80
)
