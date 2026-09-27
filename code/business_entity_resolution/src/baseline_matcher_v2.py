import pandas as pd
import re
import unicodedata
import gc
from collections import Counter, defaultdict

from rapidfuzz import fuzz


# =========================================================
# SETTINGS
# =========================================================

NAME_FREQ_LIMIT = 100
ADDRESS_FREQ_LIMIT = 100

MATCH_THRESHOLD = 0.80

CHUNK_SIZE = 200_000


# =========================================================
# 1. TEXT NORMALIZATION
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


def text_tokens(x):

    if not x:
        return set()

    return set(
        re.findall(
            r"\w+",
            x
        )
    )


# =========================================================
# 2. ENTITY F0.5
# =========================================================

def entity_f05(true_ids, pred_ids):

    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # Singleton
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    # No prediction
    if len(pred_ids) == 0:
        return 0.0

    tp = len(
        true_ids &
        pred_ids
    )

    fp = len(
        pred_ids -
        true_ids
    )

    fn = len(
        true_ids -
        pred_ids
    )

    if tp == 0:
        return 0.0

    precision = (
        tp /
        (tp + fp)
    )

    recall = (
        tp /
        (tp + fn)
    )

    return (
        1.25 * precision * recall
        /
        (0.25 * precision + recall)
    )


# =========================================================
# 3. LOAD VALIDATION IDS
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
# 4. LOAD GROUND TRUTH
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
    s1["entity_id"].isin(
        val_ids
    )
].copy()

s1["norm_name"] = s1[
    "business_name"
].apply(normalize_text)

s1["norm_address"] = s1[
    "business_address"
].apply(normalize_text)

s1["name_tokens"] = s1[
    "norm_name"
].apply(text_tokens)

s1["address_tokens"] = s1[
    "norm_address"
].apply(text_tokens)

print(
    "Validation S1:",
    len(s1)
)


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
# 8. COMBINE TARGETS
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
# 9. NORMALIZE TARGETS
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
# 10. TARGET ARRAYS
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
# 12. NAME TOKEN FREQUENCY
# =========================================================

print("Counting name token frequencies...")

name_token_counts = defaultdict(Counter)

for i in range(len(targets)):

    country = target_countries[i]
    name = target_names[i]

    for token in text_tokens(name):

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
# 13. ADDRESS TOKEN FREQUENCY
# =========================================================

print("Counting address token frequencies...")

address_token_counts = defaultdict(Counter)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    for token in text_tokens(address):

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
# 14. RARE NAME INDEX
# =========================================================

print("Building rare NAME index...")

rare_name_index = defaultdict(list)

for i in range(len(targets)):

    country = target_countries[i]
    name = target_names[i]

    counter = name_token_counts[
        country
    ]

    for token in text_tokens(name):

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
# 15. RARE ADDRESS INDEX
# =========================================================

print("Building rare ADDRESS index...")

rare_address_index = defaultdict(list)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    counter = address_token_counts[
        country
    ]

    for token in text_tokens(address):

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
# 16. RUN MATCHER
# =========================================================

print("\nStarting V2 matcher...")

total_true = 0
blocked_true = 0
true_positives = 0

total_predictions = 0
false_positives = 0

total_f05 = 0.0

singleton_count = 0
singleton_false_positive = 0

processed = 0


for row in s1.itertuples(index=False):

    country = row.country
    s1_id = row.entity_id

    # -----------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------

    candidate_indices = set()

    # =====================================================
    # BLOCK A
    # Exact normalized name + country
    # =====================================================

    candidate_indices.update(
        name_index.get(
            (
                country,
                row.norm_name
            ),
            []
        )
    )


    # =====================================================
    # BLOCK B
    # Rare address token <=100 + country
    # =====================================================

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


    # =====================================================
    # BLOCK C — NEW
    # Rare NAME token <=100 + country
    # =====================================================

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

    total_true += len(
        true_ids
    )

    if len(true_ids) == 0:
        singleton_count += 1


    # -----------------------------------------------------
    # Candidate recall
    # -----------------------------------------------------

    candidate_ids = {
        target_ids[i]
        for i in candidate_indices
    }

    blocked_true += len(
        candidate_ids &
        true_ids
    )


    # -----------------------------------------------------
    # Score candidates
    # -----------------------------------------------------

    predicted_ids = set()

    for idx in candidate_indices:

        name_score = (
            fuzz.token_set_ratio(
                row.norm_name,
                target_names[idx]
            ) / 100.0
        )

        address_score = (
            fuzz.token_set_ratio(
                row.norm_address,
                target_addresses[idx]
            ) / 100.0
        )

        # SAME matcher as Submission #1
        combined_score = (
            0.40 * name_score
            +
            0.60 * address_score
        )

        if combined_score >= MATCH_THRESHOLD:

            predicted_ids.add(
                target_ids[idx]
            )


    # -----------------------------------------------------
    # Entity statistics
    # -----------------------------------------------------

    total_predictions += len(
        predicted_ids
    )

    tp = len(
        true_ids &
        predicted_ids
    )

    fp = len(
        predicted_ids -
        true_ids
    )

    true_positives += tp
    false_positives += fp

    if (
        len(true_ids) == 0
        and len(predicted_ids) > 0
    ):

        singleton_false_positive += 1


    total_f05 += entity_f05(
        true_ids,
        predicted_ids
    )


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    processed += 1

    if processed % 10000 == 0:

        print(
            f"Processed "
            f"{processed:,} / "
            f"{len(s1):,}"
        )


# =========================================================
# 17. FINAL RESULTS
# =========================================================

macro_f05 = (
    total_f05 /
    len(s1)
)

blocking_recall = (
    blocked_true /
    total_true
)

matcher_recall = (
    true_positives /
    total_true
)


print("\n" + "=" * 85)
print("SUBMISSION #2 LOCAL BASELINE")
print("=" * 85)

print(
    "Validation S1:",
    len(s1)
)

print(
    "Total true matches:",
    total_true
)

print(
    "True matches surviving blocking:",
    blocked_true
)

print(
    "Blocking recall:",
    blocking_recall
)

print(
    "True positives:",
    true_positives
)

print(
    "Matcher recall:",
    matcher_recall
)

print(
    "Total predicted matches:",
    total_predictions
)

print(
    "False positives:",
    false_positives
)

print(
    "Singletons:",
    singleton_count
)

print(
    "Singletons with false positives:",
    singleton_false_positive
)

print(
    "Macro F0.5:",
    macro_f05
)

print("\nComparison with Submission #1:")
print("Submission #1 local F0.5: 0.558710")
print(
    "V2 local F0.5:",
    macro_f05
)

print(
    "Change:",
    macro_f05 - 0.558710
)