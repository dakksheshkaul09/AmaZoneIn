import pandas as pd
import re
import unicodedata
import gc
from collections import Counter, defaultdict
from rapidfuzz import fuzz


# =========================================================
# SETTINGS — SAME AS SUBMISSION #1
# =========================================================

ADDRESS_TOKEN_MAX_FREQ = 100
MATCH_THRESHOLD = 0.80
TOP_EXAMPLES = 30


# =========================================================
# 1. TEXT NORMALIZATION
# =========================================================

def normalize_text(x):

    if pd.isna(x):
        return ""

    x = str(x).casefold()

    # Unicode normalization
    x = unicodedata.normalize("NFKD", x)

    # Remove accents
    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )

    # Replace punctuation with spaces
    x = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in x
    )

    # Remove extra spaces
    return " ".join(x.split())


def text_tokens(x):

    if not x:
        return set()

    return set(
        re.findall(r"\w+", x)
    )


# =========================================================
# 2. F0.5 FOR ONE SOURCE 1 ENTITY
# =========================================================

def entity_f05(true_ids, pred_ids):

    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # True singleton
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
# 3. LOAD VALIDATION IDS
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
# 9. NORMALIZE TARGETS
# =========================================================

print("Normalizing targets...")

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


# Fast target ID -> row index lookup
target_id_to_index = {
    target_ids[i]: i
    for i in range(len(target_ids))
}


# =========================================================
# 11. BUILD EXACT NAME INDEX
# =========================================================

print("Building name index...")

name_index = defaultdict(list)

for i in range(len(targets)):

    key = (
        target_countries[i],
        target_names[i]
    )

    name_index[key].append(i)

gc.collect()


# =========================================================
# 12. COUNT ADDRESS TOKEN FREQUENCIES
# =========================================================

print("Counting address tokens...")

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
            f"Counted {i + 1:,} / "
            f"{len(targets):,}"
        )


# =========================================================
# 13. BUILD RARE ADDRESS INDEX
# =========================================================

print("Building address index...")

address_index = defaultdict(list)

for i in range(len(targets)):

    country = target_countries[i]
    address = target_addresses[i]

    counter = country_token_counts[country]

    for token in text_tokens(address):

        if counter[token] <= ADDRESS_TOKEN_MAX_FREQ:

            address_index[
                (country, token)
            ].append(i)

    if (i + 1) % 500000 == 0:

        print(
            f"Indexed {i + 1:,} / "
            f"{len(targets):,}"
        )

gc.collect()


# =========================================================
# 14. ERROR STATISTICS
# =========================================================

total_true = 0

# True matches that survived BLOCKING
blocked_true_matches = 0

# True matches that survived BOTH blocking and threshold
model_true_positives = 0

total_predictions = 0
false_positives = 0

total_f05 = 0.0

singleton_count = 0
singleton_false_positive = 0

true_score_bins = Counter()
false_score_bins = Counter()

missed_true_examples = []
low_true_examples = []
false_positive_examples = []

processed = 0


# =========================================================
# 15. PROCESS VALIDATION S1
# =========================================================

print("\nAnalyzing matcher errors...")

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

        if counter.get(
            token,
            0
        ) <= ADDRESS_TOKEN_MAX_FREQ:

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

    total_true += len(true_ids)

    if len(true_ids) == 0:
        singleton_count += 1

    # -----------------------------------------------------
    # ACTUAL BLOCKING RECALL
    # -----------------------------------------------------

    candidate_ids = {
        target_ids[idx]
        for idx in candidate_indices
    }

    blocked_true = (
        true_ids &
        candidate_ids
    )

    blocked_true_matches += len(
        blocked_true
    )

    # -----------------------------------------------------
    # Score every candidate
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

        combined_score = (
            0.40 * name_score
            +
            0.60 * address_score
        )

        # Score bin
        score_bin = round(
            combined_score * 20
        ) / 20

        is_true = (
            target_ids[idx]
            in true_ids
        )

        # -------------------------------------------------
        # Score distributions
        # -------------------------------------------------

        if is_true:

            true_score_bins[
                score_bin
            ] += 1

        else:

            false_score_bins[
                score_bin
            ] += 1

        # -------------------------------------------------
        # Prediction decision
        # -------------------------------------------------

        if combined_score >= MATCH_THRESHOLD:

            # EVERY candidate above threshold
            # becomes a prediction
            predicted_ids.add(
                target_ids[idx]
            )

            if is_true:

                model_true_positives += 1

            else:

                false_positive_examples.append(
                    {
                        "s1_id": s1_id,
                        "s1_name": row.business_name,
                        "s1_address": row.business_address,
                        "target_id": target_ids[idx],
                        "target_name": targets.iloc[idx]["business_name"],
                        "target_address": targets.iloc[idx]["business_address"],
                        "name_score": name_score,
                        "address_score": address_score,
                        "combined_score": combined_score
                    }
                )

        # -------------------------------------------------
        # True match blocked but rejected by matcher
        # -------------------------------------------------

        elif is_true:

            low_true_examples.append(
                {
                    "s1_id": s1_id,
                    "s1_name": row.business_name,
                    "s1_address": row.business_address,
                    "target_id": target_ids[idx],
                    "target_name": targets.iloc[idx]["business_name"],
                    "target_address": targets.iloc[idx]["business_address"],
                    "name_score": name_score,
                    "address_score": address_score,
                    "combined_score": combined_score
                }
            )

    # -----------------------------------------------------
    # Entity-level prediction statistics
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

    true_positives_for_entity = tp

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
    # TRUE MATCHES MISSED BY BLOCKING
    # -----------------------------------------------------

    missed_ids = (
        true_ids -
        candidate_ids
    )

    for true_id in missed_ids:

        if len(missed_true_examples) >= TOP_EXAMPLES:
            break

        target_idx = target_id_to_index.get(
            true_id
        )

        if target_idx is not None:

            missed_true_examples.append(
                {
                    "s1_id": s1_id,
                    "s1_name": row.business_name,
                    "s1_address": row.business_address,
                    "target_id": true_id,
                    "target_name": targets.iloc[target_idx]["business_name"],
                    "target_address": targets.iloc[target_idx]["business_address"]
                }
            )

    processed += 1

    if processed % 10000 == 0:

        print(
            f"Processed {processed:,} / "
            f"{len(s1):,}"
        )


# =========================================================
# 16. FINAL METRICS
# =========================================================

macro_f05 = (
    total_f05 /
    len(s1)
)

blocking_recall = (
    blocked_true_matches /
    total_true
)

matcher_recall = (
    model_true_positives /
    total_true
)


# =========================================================
# 17. PRINT SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("BASELINE ERROR ANALYSIS")
print("=" * 80)

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
    blocked_true_matches
)

print(
    "ACTUAL BLOCKING RECALL:",
    blocking_recall
)

print(
    "True positives after matcher:",
    model_true_positives
)

print(
    "MATCHER RECALL:",
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


# =========================================================
# 18. TRUE MATCH SCORE DISTRIBUTION
# =========================================================

print("\n" + "=" * 80)
print("TRUE MATCH SCORE DISTRIBUTION")
print("=" * 80)

for score_bin in sorted(true_score_bins):

    print(
        f"{score_bin:.2f}: "
        f"{true_score_bins[score_bin]}"
    )


# =========================================================
# 19. FALSE CANDIDATE SCORE DISTRIBUTION
# =========================================================

print("\n" + "=" * 80)
print("FALSE CANDIDATE SCORE DISTRIBUTION")
print("=" * 80)

for score_bin in sorted(false_score_bins):

    print(
        f"{score_bin:.2f}: "
        f"{false_score_bins[score_bin]}"
    )


# =========================================================
# 20. SAVE ERROR EXAMPLES
# =========================================================

pd.DataFrame(
    low_true_examples
).sort_values(
    "combined_score"
).to_csv(
    "low_scoring_true_examples.tsv",
    sep="\t",
    index=False
)


pd.DataFrame(
    false_positive_examples
).sort_values(
    "combined_score",
    ascending=False
).head(
    TOP_EXAMPLES
).to_csv(
    "top_false_positive_examples.tsv",
    sep="\t",
    index=False
)


pd.DataFrame(
    missed_true_examples
).to_csv(
    "missed_true_examples.tsv",
    sep="\t",
    index=False
)


# =========================================================
# 21. DONE
# =========================================================

print("\nSaved:")
print("low_scoring_true_examples.tsv")
print("top_false_positive_examples.tsv")
print("missed_true_examples.tsv")