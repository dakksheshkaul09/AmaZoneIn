import pandas as pd
import re
import unicodedata
import gc
from collections import Counter, defaultdict


# =========================================================
# SETTINGS
# =========================================================

ADDRESS_FREQ_LIMIT = 100

NAME_THRESHOLDS = [
    100,
    500,
    1000
]

CHUNK_SIZE = 200_000


# =========================================================
# 1. NORMALIZATION
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


def name_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(
            r"\w+",
            str(x).casefold()
        )
    )


def address_tokens(x):

    # IMPORTANT:
    # This matches our Submission #1 logic:
    # normalize address FIRST, then tokenize.

    x = normalize_text(x)

    if not x:
        return set()

    return set(
        re.findall(
            r"\w+",
            x
        )
    )


# =========================================================
# 2. LOAD VALIDATION IDS
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
# 3. LOAD VALIDATION SOURCE 1
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


# Create compact lookup:
# S1 ID -> information we need

s1_map = {}

for row in s1.itertuples(index=False):

    s1_map[
        row.entity_id
    ] = {
        "country": row.country,
        "norm_name": normalize_text(
            row.business_name
        ),
        "name_tokens": name_tokens(
            row.business_name
        ),
        "address_tokens": address_tokens(
            row.business_address
        )
    }


print(
    "Validation S1:",
    len(s1_map)
)

del s1
del val_ids
gc.collect()


# =========================================================
# 4. LOAD VALIDATION GROUND TRUTH
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

# Remove singletons
gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
].copy()


# =========================================================
# 5. BUILD TARGET ID -> S1 ID MAP
#
# We already discovered each S2/S3 ID belongs
# to at most one S1 in the ground truth.
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
        "matched_entity_ids": "target_id"
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
# 6. FIRST PASS:
# COUNT NAME + ADDRESS TOKEN FREQUENCIES
#
# We NEVER store token sets for the 10M rows.
# =========================================================

print("\nFIRST PASS: counting token frequencies...")

name_counts = defaultdict(Counter)
address_counts = defaultdict(Counter)


def count_file(path):

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
            # Name tokens
            # -----------------------------

            for token in name_tokens(
                row.business_name
            ):

                name_counts[
                    country
                ][token] += 1


            # -----------------------------
            # Address tokens
            # -----------------------------

            for token in address_tokens(
                row.business_address
            ):

                address_counts[
                    country
                ][token] += 1


        processed += len(chunk)

        print(
            f"Processed {processed:,}"
        )

        del chunk
        gc.collect()


count_file(
    "train_source2.tsv"
)

count_file(
    "train_source3.tsv"
)


print("\nFirst pass complete.")


# =========================================================
# 7. SECOND PASS:
# INSPECT ONLY TRUE VALIDATION TARGETS
#
# No giant DataFrame.
# =========================================================

print("\nSECOND PASS: analyzing missed true matches...")


missed_total = 0

name_recovery = {
    threshold: 0
    for threshold in NAME_THRESHOLDS
}

missed_with_any_shared_name = 0


def analyze_file(path):

    global missed_total
    global missed_with_any_shared_name

    print("\nScanning:", path)

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

            # We only care about true validation targets
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
            # Reproduce current Submission #1 blocking logic
            # =================================================

            current_blocked = False

            # ---------------------------------------------
            # Block A: exact normalized name + country
            # ---------------------------------------------

            target_norm_name = normalize_text(
                row.business_name
            )

            if (
                country == s1_row["country"]
                and
                target_norm_name ==
                s1_row["norm_name"]
            ):

                current_blocked = True


            # ---------------------------------------------
            # Block B: rare normalized-address token
            # + country
            # ---------------------------------------------

            if not current_blocked:

                target_address_tokens = (
                    address_tokens(
                        row.business_address
                    )
                )

                if country == s1_row["country"]:

                    counter = address_counts[
                        country
                    ]

                    shared_address = (
                        s1_row["address_tokens"]
                        &
                        target_address_tokens
                    )

                    for token in shared_address:

                        if counter.get(
                            token,
                            0
                        ) <= ADDRESS_FREQ_LIMIT:

                            current_blocked = True
                            break


            # =================================================
            # We only care about TRUE matches missed by
            # current blocker.
            # =================================================

            if current_blocked:
                continue


            missed_total += 1

            # =================================================
            # Test NAME token overlap
            # =================================================

            target_name_token_set = (
                name_tokens(
                    row.business_name
                )
            )

            shared_name_tokens = (
                s1_row["name_tokens"]
                &
                target_name_token_set
            )

            if not shared_name_tokens:
                continue

            missed_with_any_shared_name += 1

            counter = name_counts[
                country
            ]

            rarest_frequency = min(
                counter.get(
                    token,
                    float("inf")
                )
                for token in shared_name_tokens
            )

            for threshold in NAME_THRESHOLDS:

                if (
                    rarest_frequency
                    <= threshold
                ):

                    name_recovery[
                        threshold
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
# 8. RESULTS
# =========================================================

print("\n" + "=" * 80)

print(
    "RARE NAME TOKEN COVERAGE ON CURRENTLY MISSED TRUE MATCHES"
)

print("=" * 80)

print(
    "Current blocker missed true matches:",
    missed_total
)

print(
    "Missed matches sharing >=1 name token:",
    missed_with_any_shared_name
)

if missed_total > 0:

    print(
        "Percentage sharing >=1 name token:",
        missed_with_any_shared_name
        / missed_total
    )


print("\nRare name-token recovery:")

for threshold in NAME_THRESHOLDS:

    recovered = name_recovery[
        threshold
    ]

    coverage = (
        recovered / missed_total
        if missed_total > 0
        else 0
    )

    print(
        f"Name token frequency <= "
        f"{threshold:4}: "
        f"{recovered:8} recovered | "
        f"{coverage:7.2%}"
    )