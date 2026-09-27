import pandas as pd
import re
import unicodedata
import gc
from collections import Counter, defaultdict


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
# 2. LEGAL SUFFIX CANONICALIZATION
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

    """
    Canonicalize common legal suffixes.

    Example:
    Pvt. Ltd. -> private limited
    Corp -> corporation
    Inc -> incorporated
    """

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

    """
    Remove generic legal suffixes and sort the
    remaining tokens.

    This captures cases such as:

    'ABC Pvt Ltd'
    'ABC Limited Private'

    -> 'abc'
    """

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
# 3. ADDRESS CANONICALIZATION
# =========================================================

ADDRESS_MAP = {

    # Roads
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

    # Building/unit
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


def strong_address_tokens(x):

    """
    Normalize address tokens and sort them.

    Sorting makes this representation insensitive
    to address component order.
    """

    x = normalize_text(x)

    if not x:
        return tuple()

    result = []

    for token in x.split():

        result.append(
            ADDRESS_MAP.get(
                token,
                token
            )
        )

    # set() removes duplicate tokens
    # sorted() removes ordering sensitivity
    return tuple(
        sorted(
            set(result)
        )
    )


def strong_address_string(x):

    return " ".join(
        strong_address_tokens(x)
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
    val_ids_df[
        "source1_entity_id"
    ]
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


# We store ONLY one compact dictionary.

s1_map = {}

for row in s1.itertuples(index=False):

    s1_map[
        row.entity_id
    ] = {

        "country":
            row.country,

        # Old representations
        "old_name":
            normalize_text(
                row.business_name
            ),

        "old_name_tokens":
            tokens(
                normalize_text(
                    row.business_name
                )
            ),

        "old_address_tokens":
            tokens(
                normalize_text(
                    row.business_address
                )
            ),

        # Strong representations
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
# 6. LOAD GROUND TRUTH
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
# 7. TRUE TARGET -> SOURCE 1 LOOKUP
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
# 8. PASS 1
#
# Count name/address token frequencies exactly
# like our current V2 blocker.
# =========================================================

print("\n" + "=" * 70)
print("PASS 1 — COUNTING OLD BLOCKER TOKENS")
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

        for row in chunk.itertuples(
            index=False
        ):

            country = row.country

            # Old name normalization
            old_name = normalize_text(
                row.business_name
            )

            for token in tokens(
                old_name
            ):

                name_counts[
                    country
                ][token] += 1

            # Old address normalization
            old_address = normalize_text(
                row.business_address
            )

            for token in tokens(
                old_address
            ):

                address_counts[
                    country
                ][token] += 1

        processed += len(chunk)

        if processed % 1_000_000 == 0:

            print(
                f"Processed {processed:,}"
            )

        del chunk

    gc.collect()


count_tokens(
    "train_source2.tsv"
)

count_tokens(
    "train_source3.tsv"
)

print("\nPASS 1 COMPLETE")


# =========================================================
# 9. STATISTICS
# =========================================================

current_blocked = 0

strong_name_full_count = 0
strong_name_core_count = 0
strong_address_count = 0

strong_name_full_incremental = 0
strong_name_core_incremental = 0
strong_address_incremental = 0

strong_any_incremental = 0

total_pairs = len(
    target_to_s1
)


# =========================================================
# 10. PASS 2
#
# Look ONLY at genuine validation targets.
# =========================================================

print("\n" + "=" * 70)
print("PASS 2 — ANALYZING TRUE VALIDATION PAIRS")
print("=" * 70)


def analyze_file(path):

    global current_blocked

    global strong_name_full_count
    global strong_name_core_count
    global strong_address_count

    global strong_name_full_incremental
    global strong_name_core_incremental
    global strong_address_incremental
    global strong_any_incremental

    processed = 0
    relevant = 0

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

            # We only care about true validation targets.
            if target_id not in target_to_s1:
                continue

            relevant += 1

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
            # Current Block A:
            # exact normalized name + country
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
            # Current Block B:
            # rare address token <=100 + country
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

                        if counter.get(
                            token,
                            0
                        ) <= ADDRESS_FREQ_LIMIT:

                            blocked = True
                            break


            # -------------------------------------------------
            # Current Block C:
            # rare name token <=100 + country
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

                        if counter.get(
                            token,
                            0
                        ) <= NAME_FREQ_LIMIT:

                            blocked = True
                            break


            if blocked:

                current_blocked += 1


            # =================================================
            # STRONG REPRESENTATIONS
            # =================================================

            new_name = strong_name(
                row.business_name
            )

            new_name_core = strong_name_core(
                row.business_name
            )

            new_address = strong_address_string(
                row.business_address
            )


            name_full_match = (
                country ==
                s1_row["country"]
                and
                new_name ==
                s1_row["strong_name"]
            )

            name_core_match = (
                country ==
                s1_row["country"]
                and
                new_name_core ==
                s1_row["strong_name_core"]
                and
                new_name_core != ""
            )

            address_match = (
                country ==
                s1_row["country"]
                and
                new_address ==
                s1_row["strong_address"]
                and
                new_address != ""
            )


            # -------------------------------------------------
            # Overall strong exact signals
            # -------------------------------------------------

            if name_full_match:
                strong_name_full_count += 1

            if name_core_match:
                strong_name_core_count += 1

            if address_match:
                strong_address_count += 1


            # -------------------------------------------------
            # IMPORTANT:
            # Only count INCREMENTAL recovery if the
            # current V2 blocker missed this true match.
            # -------------------------------------------------

            if not blocked:

                if name_full_match:

                    strong_name_full_incremental += 1

                if name_core_match:

                    strong_name_core_incremental += 1

                if address_match:

                    strong_address_incremental += 1

                if (
                    name_full_match
                    or
                    name_core_match
                    or
                    address_match
                ):

                    strong_any_incremental += 1


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
print("STRONG NORMALIZATION RESULTS")
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
    "Current V2 blocking recall:",
    current_blocked / total_pairs
)

print("\n--- Strong signal over ALL true pairs ---")

print(
    "Strong full name exact:",
    strong_name_full_count,
    f"({strong_name_full_count / total_pairs:.2%})"
)

print(
    "Strong core name exact:",
    strong_name_core_count,
    f"({strong_name_core_count / total_pairs:.2%})"
)

print(
    "Strong unordered address exact:",
    strong_address_count,
    f"({strong_address_count / total_pairs:.2%})"
)

print("\n--- Incremental recovery of CURRENT V2 misses ---")

missed_current = (
    total_pairs -
    current_blocked
)

print(
    "Current V2 misses:",
    missed_current
)

print(
    "Strong full name recovers:",
    strong_name_full_incremental,
    f"({strong_name_full_incremental / missed_current:.2%})"
)

print(
    "Strong core name recovers:",
    strong_name_core_incremental,
    f"({strong_name_core_incremental / missed_current:.2%})"
)

print(
    "Strong address recovers:",
    strong_address_incremental,
    f"({strong_address_incremental / missed_current:.2%})"
)

print(
    "ANY strong signal recovers:",
    strong_any_incremental,
    f"({strong_any_incremental / missed_current:.2%})"
)


# =========================================================
# 12. POTENTIAL NEW BLOCKING CEILING
# =========================================================

potential_recovered = (
    current_blocked +
    strong_any_incremental
)

potential_recall = (
    potential_recovered /
    total_pairs
)

print("\n" + "=" * 80)
print("POTENTIAL RECALL CEILING")
print("=" * 80)

print(
    "Current V2 recall:",
    current_blocked / total_pairs
)

print(
    "Current V2 + ANY strong signal:",
    potential_recovered,
    "/",
    total_pairs
)

print(
    "Potential recall:",
    potential_recall
)

print(
    "Potential gain:",
    potential_recall -
    (current_blocked / total_pairs)
)