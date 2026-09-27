import pandas as pd
import re
import unicodedata
import gc

from collections import Counter, defaultdict


# =========================================================
# SETTINGS
# =========================================================

CHUNK_SIZE = 200_000

# Only use character 3-grams whose global frequency
# is at most this value.
MAX_GRAM_FREQS = [
    1000,
    5000
]

# Number of rarest grams selected from each S1
TOP_K_VALUES = [
    1,
    2,
    3,
    5,
    10
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
# 4. CHARACTER 3-GRAMS
# =========================================================

def char_ngrams(x):

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
    s1["entity_id"].isin(
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

    strong_address = strong_address_string(
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
                core_name
            ),

        "address_ngrams":
            char_ngrams(
                strong_address
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
# 8. TRUE TARGET -> S1
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
# 9. COUNT OLD V2 TOKENS
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
# 10. COUNT CHARACTER 3-GRAM FREQUENCIES
# =========================================================

print("\n" + "=" * 70)
print("PASS 1B — CHARACTER 3-GRAM FREQUENCIES")
print("=" * 70)

name_ngram_counts = defaultdict(Counter)
address_ngram_counts = defaultdict(Counter)


def count_ngrams(path):

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


count_ngrams(
    "train_source2.tsv"
)

count_ngrams(
    "train_source3.tsv"
)


# =========================================================
# 11. SELECT TOP-K RAREST GRAMS PER S1
# =========================================================

print("\n" + "=" * 70)
print("BUILDING TOP-K RARE GRAM BLOCKERS")
print("=" * 70)


selected_grams = {}


for s1_id, row in s1_map.items():

    country = row["country"]

    name_counter = (
        name_ngram_counts[
            country
        ]
    )

    address_counter = (
        address_ngram_counts[
            country
        ]
    )


    selected_grams[s1_id] = {}


    for max_freq in MAX_GRAM_FREQS:

        selected_grams[
            s1_id
        ][
            max_freq
        ] = {}


        for k in TOP_K_VALUES:

            # ---------------------------------------------
            # Name
            # ---------------------------------------------

            eligible_name = []

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
                    freq <= max_freq
                ):

                    eligible_name.append(
                        (
                            freq,
                            gram
                        )
                    )


            eligible_name.sort()

            top_name = {
                gram
                for freq, gram
                in eligible_name[:k]
            }


            # ---------------------------------------------
            # Address
            # ---------------------------------------------

            eligible_address = []

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
                    freq <= max_freq
                ):

                    eligible_address.append(
                        (
                            freq,
                            gram
                        )
                    )


            eligible_address.sort()

            top_address = {
                gram
                for freq, gram
                in eligible_address[:k]
            }


            selected_grams[
                s1_id
            ][
                max_freq
            ][
                k
            ] = {

                "name":
                    top_name,

                "address":
                    top_address
            }


# =========================================================
# 12. CANDIDATE UPPER BOUNDS
# =========================================================

print("\n" + "=" * 70)
print("CANDIDATE UPPER-BOUND ESTIMATES")
print("=" * 70)


for max_freq in MAX_GRAM_FREQS:

    name_values = []
    address_values = []
    combined_values = []


    for s1_id, row in s1_map.items():

        name_counter = (
            name_ngram_counts[
                row["country"]
            ]
        )

        address_counter = (
            address_ngram_counts[
                row["country"]
            ]
        )


        for k in TOP_K_VALUES:

            name_grams = selected_grams[
                s1_id
            ][
                max_freq
            ][
                k
            ][
                "name"
            ]

            address_grams = selected_grams[
                s1_id
            ][
                max_freq
            ][
                k
            ][
                "address"
            ]


            name_sum = sum(
                name_counter.get(
                    gram,
                    0
                )
                for gram
                in name_grams
            )

            address_sum = sum(
                address_counter.get(
                    gram,
                    0
                )
                for gram
                in address_grams
            )


            name_values.append(
                (
                    k,
                    name_sum
                )
            )

            address_values.append(
                (
                    k,
                    address_sum
                ))


    print(
        f"\nRare frequency <= {max_freq}"
    )


    for k in TOP_K_VALUES:

        n_values = [
            value
            for kk, value
            in name_values
            if kk == k
        ]

        a_values = [
            value
            for kk, value
            in address_values
            if kk == k
        ]

        total_values = [
            n + a
            for n, a
            in zip(
                n_values,
                a_values
            )
        ]


        def percentile(values, p):

            if not values:
                return 0

            values = sorted(
                values
            )

            index = int(
                len(values) * p
            )

            if index >= len(values):
                index = len(values) - 1

            return values[
                index
            ]


        print(
            f"\nTOP {k}"
        )

        print(
            "  Name avg upper bound:",
            sum(n_values) /
            len(n_values)
        )

        print(
            "  Name P95:",
            percentile(
                n_values,
                0.95
            )
        )

        print(
            "  Name P99:",
            percentile(
                n_values,
                0.99
            )
        )

        print(
            "  Address avg upper bound:",
            sum(a_values) /
            len(a_values)
        )

        print(
            "  Address P95:",
            percentile(
                a_values,
                0.95
            )
        )

        print(
            "  Address P99:",
            percentile(
                a_values,
                0.99
            )
        )

        print(
            "  Combined avg upper bound:",
            sum(total_values) /
            len(total_values)
        )

        print(
            "  Combined P95:",
            percentile(
                total_values,
                0.95
            )
        )

        print(
            "  Combined P99:",
            percentile(
                total_values,
                0.99
            )
        )


# =========================================================
# 13. CEILING ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TOP-K RARE 3-GRAM RECALL CEILING")
print("=" * 70)


total_pairs = len(
    target_to_s1
)

current_v2_blocked = 0
current_v2_misses = 0


recoveries = {

    max_freq: {

        k: 0

        for k in TOP_K_VALUES

    }

    for max_freq
    in MAX_GRAM_FREQS
}


def analyze_file(path):

    global current_v2_blocked
    global current_v2_misses

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


            # Block A: exact name

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


            # Block C: rare name token

            if not blocked:

                if country == s1_row["country"]:

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

                current_v2_blocked += 1

                continue


            current_v2_misses += 1


            # =================================================
            # TARGET STRONG REPRESENTATIONS
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


            # =================================================
            # TEST TOP-K BLOCKERS
            # =================================================

            for max_freq in MAX_GRAM_FREQS:

                for k in TOP_K_VALUES:

                    config = selected_grams[
                        s1_id
                    ][
                        max_freq
                    ][
                        k
                    ]


                    name_hit = bool(
                        config[
                            "name"
                        ]
                        &
                        target_name_grams
                    )


                    address_hit = bool(
                        config[
                            "address"
                        ]
                        &
                        target_address_grams
                    )


                    if (
                        name_hit
                        or
                        address_hit
                    ):

                        recoveries[
                            max_freq
                        ][
                            k
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
# 14. RESULTS
# =========================================================

print("\n" + "=" * 80)
print("FINAL TOP-K RESULTS")
print("=" * 80)

print(
    "Total true validation pairs:",
    total_pairs
)

print(
    "Current V2 blocked:",
    current_v2_blocked
)

print(
    "Current V2 misses:",
    current_v2_misses
)


for max_freq in MAX_GRAM_FREQS:

    print(
        "\n" + "-" * 70
    )

    print(
        f"RARE FREQUENCY <= {max_freq}"
    )

    for k in TOP_K_VALUES:

        recovered = recoveries[
            max_freq
        ][
            k
        ]

        total_recovered = (
            current_v2_blocked
            +
            recovered
        )

        recall = (
            total_recovered /
            total_pairs
        )

        miss_recovery = (
            recovered /
            current_v2_misses
        )


        print(
            f"TOP {k}: "
            f"{recovered:,} additional | "
            f"{miss_recovery:.2%} of V2 misses | "
            f"total recall {recall:.2%}"
        )


print("\nDONE.")