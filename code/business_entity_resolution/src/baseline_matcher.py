import pandas as pd
import re
import unicodedata
import gc
from collections import Counter, defaultdict

from rapidfuzz import fuzz


# =========================================================
# 1. NORMALIZATION
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


def text_tokens(x):

    if not x:
        return set()

    return set(
        re.findall(r"\w+", x)
    )


# =========================================================
# 2. F0.5
# =========================================================

def entity_f05(true_ids, pred_ids):

    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # Singleton
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    # No predictions
    if len(pred_ids) == 0:
        return 0.0

    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


# =========================================================
# 3. LOAD VALIDATION IDs
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
gc.collect()


# =========================================================
# 4. LOAD GROUND TRUTH
# =========================================================

print("Loading ground truth...")

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(val_ids)
].copy()

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].fillna("")
    )
)

del gt
gc.collect()


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

s1["norm_name"] = s1[
    "business_name"
].apply(normalize_text)

s1["norm_address"] = s1[
    "business_address"
].apply(normalize_text)

print(
    "Validation S1:",
    len(s1)
)

del val_ids
gc.collect()


# =========================================================
# 6. LOAD SOURCE 2
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


# =========================================================
# 7. LOAD SOURCE 3
# =========================================================

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


# =========================================================
# 8. COMBINE TARGET DATA
# =========================================================

print("Combining Source 2 + Source 3...")

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
# 9. NORMALIZE TARGET NAME + ADDRESS
#
# IMPORTANT:
# We DO NOT create token columns here.
# =========================================================

print("Normalizing target data...")

targets["norm_name"] = targets[
    "business_name"
].apply(normalize_text)

targets["norm_address"] = targets[
    "business_address"
].apply(normalize_text)

gc.collect()


# =========================================================
# 10. PREPARE TARGET ARRAYS
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
# 11. EXACT NAME INDEX
# =========================================================

print("Building exact-name index...")

name_index = defaultdict(list)

for i in range(len(targets)):

    key = (
        target_countries[i],
        target_names[i]
    )

    name_index[key].append(i)

gc.collect()


# =========================================================
# 12. ADDRESS TOKEN FREQUENCY
#
# We tokenize ONE address at a time.
# Nothing is stored back into the DataFrame.
# =========================================================

print("Counting address token frequencies...")

country_token_counts = defaultdict(Counter)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    for token in text_tokens(address):

        country_token_counts[
            country
        ][token] += 1

    if (i + 1) % 500000 == 0:
        print(
            f"Processed {i + 1:,} / "
            f"{len(targets):,} addresses"
        )


# =========================================================
# 13. ADDRESS TOKEN INDEX
#
# ONLY rare tokens (frequency <= 100)
# are added to the index.
# =========================================================

print("Building rare address-token index...")

address_index = defaultdict(list)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    counter = country_token_counts[country]

    for token in text_tokens(address):

        if counter[token] <= 100:

            address_index[
                (country, token)
            ].append(i)

    if (i + 1) % 500000 == 0:
        print(
            f"Indexed {i + 1:,} / "
            f"{len(targets):,} addresses"
        )

gc.collect()


# =========================================================
# 14. RUN MATCHER
# =========================================================

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


print("\nStarting baseline matcher...")

results = {
    threshold: []
    for threshold in thresholds
}

processed = 0


for row in s1.itertuples(index=False):

    s1_id = row.entity_id
    country = row.country

    # -----------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------

    candidate_indices = set()

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
    # Rare address token + country
    counter = country_token_counts.get(
        country,
        {}
    )

    for token in text_tokens(
        row.norm_address
    ):

        if counter.get(token, 0) <= 100:

            candidate_indices.update(
                address_index.get(
                    (
                        country,
                        token
                    ),
                    []
                )
            )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    true_string = gt_map.get(
        s1_id,
        ""
    )

    if true_string == "":
        true_ids = set()
    else:
        true_ids = set(
            true_string.split(",")
        )

    # -----------------------------------------------------
    # Score candidates
    # -----------------------------------------------------

    candidate_scores = []

    for idx in candidate_indices:

        target_name = target_names[idx]
        target_address = target_addresses[idx]

        name_score = (
            fuzz.token_set_ratio(
                row.norm_name,
                target_name
            ) / 100.0
        )

        address_score = (
            fuzz.token_set_ratio(
                row.norm_address,
                target_address
            ) / 100.0
        )

        combined_score = (
            0.40 * name_score
            +
            0.60 * address_score
        )

        candidate_scores.append(
            (
                target_ids[idx],
                combined_score
            )
        )

    # -----------------------------------------------------
    # Test all thresholds
    # -----------------------------------------------------

    for threshold in thresholds:

        predicted_ids = {
            entity_id
            for entity_id, score
            in candidate_scores
            if score >= threshold
        }

        score = entity_f05(
            true_ids,
            predicted_ids
        )

        results[
            threshold
        ].append(score)

    processed += 1

    if processed % 10000 == 0:

        print(
            f"Matched {processed:,} / "
            f"{len(s1):,}"
        )


# =========================================================
# 15. RESULTS
# =========================================================

print("\n" + "=" * 80)
print("BASELINE MATCHER RESULTS")
print("=" * 80)

print(
    f"{'Threshold':>12}"
    f"{'Macro F0.5':>18}"
)

best_threshold = None
best_score = -1


for threshold in thresholds:

    score = (
        sum(results[threshold])
        /
        len(results[threshold])
    )

    print(
        f"{threshold:>12.2f}"
        f"{score:>18.6f}"
    )

    if score > best_score:

        best_score = score
        best_threshold = threshold


print("\nBest threshold:", best_threshold)
print(
    "Best validation F0.5:",
    best_score
)