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

s1["norm_name"] = s1[
    "business_name"
].apply(normalize_text)

s1["norm_address"] = s1[
    "business_address"
].apply(normalize_text)

s1["name_tokens"] = s1[
    "norm_name"
].apply(tokens)

s1["address_tokens"] = s1[
    "norm_address"
].apply(tokens)

print(
    "Validation S1:",
    len(s1)
)


# =========================================================
# 4. LOAD S2 + S3
# =========================================================

print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

del s2
del s3
gc.collect()

print(
    "Total target records:",
    len(targets)
)


# =========================================================
# 5. NORMALIZE TARGETS
# =========================================================

print("Normalizing target data...")

targets["norm_name"] = targets[
    "business_name"
].apply(normalize_text)

targets["norm_address"] = targets[
    "business_address"
].apply(normalize_text)

# DO NOT store token sets in the dataframe.


# =========================================================
# 6. PREPARE ARRAYS
# =========================================================

target_ids = targets[
    "entity_id"
].tolist()

target_names = targets[
    "norm_name"
].tolist()

target_addresses = targets[
    "norm_address"
].tolist()

target_countries = targets[
    "country"
].tolist()


# =========================================================
# 7. EXACT NAME INDEX
# =========================================================

print("Building exact-name index...")

name_index = defaultdict(list)

for i in range(len(targets)):

    key = (
        target_countries[i],
        target_names[i]
    )

    name_index[key].append(i)


# =========================================================
# 8. NAME TOKEN FREQUENCIES
# =========================================================

print("Counting NAME token frequencies...")

name_token_counts = defaultdict(Counter)

for i in range(len(targets)):

    country = target_countries[i]
    name = target_names[i]

    for token in tokens(name):

        name_token_counts[
            country
        ][token] += 1

    if (i + 1) % 500000 == 0:

        print(
            f"Counted names "
            f"{i + 1:,} / "
            f"{len(targets):,}"
        )


# =========================================================
# 9. ADDRESS TOKEN FREQUENCIES
# =========================================================

print("Counting ADDRESS token frequencies...")

address_token_counts = defaultdict(Counter)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    for token in tokens(address):

        address_token_counts[
            country
        ][token] += 1

    if (i + 1) % 500000 == 0:

        print(
            f"Counted addresses "
            f"{i + 1:,} / "
            f"{len(targets):,}"
        )


# =========================================================
# 10. BUILD RARE NAME INDEX
# =========================================================

print("Building rare NAME-token index...")

rare_name_index = defaultdict(list)

for i in range(len(targets)):

    country = target_countries[i]
    name = target_names[i]

    counter = name_token_counts[
        country
    ]

    for token in tokens(name):

        if counter[token] <= NAME_FREQ_LIMIT:

            rare_name_index[
                (
                    country,
                    token
                )
            ].append(i)

    if (i + 1) % 500000 == 0:

        print(
            f"Indexed names "
            f"{i + 1:,} / "
            f"{len(targets):,}"
        )


# =========================================================
# 11. BUILD RARE ADDRESS INDEX
# =========================================================

print("Building rare ADDRESS-token index...")

rare_address_index = defaultdict(list)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    counter = address_token_counts[
        country
    ]

    for token in tokens(address):

        if counter[token] <= ADDRESS_FREQ_LIMIT:

            rare_address_index[
                (
                    country,
                    token
                )
            ].append(i)

    if (i + 1) % 500000 == 0:

        print(
            f"Indexed addresses "
            f"{i + 1:,} / "
            f"{len(targets):,}"
        )

gc.collect()


# =========================================================
# 12. LOAD GROUND TRUTH
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
        val_ids
    )
].copy()

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

del gt


# =========================================================
# 13. TEST UPGRADED BLOCKER
# =========================================================

print("\nTesting upgraded blocker...")

total_candidates = 0
s1_with_candidates = 0

total_true = 0
recovered_true = 0

all_true_recovered = 0


for row in s1.itertuples(index=False):

    # -----------------------------------------------------
    # Build candidate set
    # -----------------------------------------------------

    candidate_indices = set()

    country = row.country

    # BLOCK A:
    # Exact normalized name + country

    candidate_indices.update(
        name_index.get(
            (
                country,
                row.norm_name
            ),
            []
        )
    )


    # BLOCK B:
    # Rare address token <=100 + country

    address_counter = (
        address_token_counts.get(
            country,
            {}
        )
    )

    for token in row.address_tokens:

        if address_counter.get(
            token,
            0
        ) <= ADDRESS_FREQ_LIMIT:

            candidate_indices.update(
                rare_address_index.get(
                    (
                        country,
                        token
                    ),
                    []
                )
            )


    # BLOCK C:
    # Rare NAME token <=100 + country

    name_counter = (
        name_token_counts.get(
            country,
            {}
        )
    )

    for token in row.name_tokens:

        if name_counter.get(
            token,
            0
        ) <= NAME_FREQ_LIMIT:

            candidate_indices.update(
                rare_name_index.get(
                    (
                        country,
                        token
                    ),
                    []
                )
            )


    # -----------------------------------------------------
    # Candidate statistics
    # -----------------------------------------------------

    count = len(
        candidate_indices
    )

    total_candidates += count

    if count > 0:
        s1_with_candidates += 1


    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    true_string = gt_map.get(
        row.entity_id,
        ""
    )

    if true_string == "":

        true_ids = set()

    else:

        true_ids = set(
            true_string.split(",")
        )


    total_true += len(true_ids)


    # -----------------------------------------------------
    # Candidate recall
    # -----------------------------------------------------

    candidate_ids = {
        target_ids[i]
        for i in candidate_indices
    }

    recovered = (
        true_ids &
        candidate_ids
    )

    recovered_true += len(
        recovered
    )

    if (
        len(recovered)
        ==
        len(true_ids)
    ):

        all_true_recovered += 1


# =========================================================
# 14. RESULTS
# =========================================================

print("\n" + "=" * 85)
print("BLOCKER V2 RESULTS")
print("=" * 85)

print(
    "Validation S1:",
    len(s1)
)

print(
    "S1 with >=1 candidate:",
    s1_with_candidates
)

print(
    "Total candidates:",
    total_candidates
)

print(
    "Average candidates/S1:",
    total_candidates / len(s1)
)

print(
    "Total true matches:",
    total_true
)

print(
    "Recovered true matches:",
    recovered_true
)

print(
    "Candidate recall:",
    recovered_true / total_true
)

print(
    "S1 with ALL true matches recovered:",
    all_true_recovered
)