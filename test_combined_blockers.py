import pandas as pd
import re
import unicodedata
from collections import Counter, defaultdict


# =========================================================
# 1. Normalization
# =========================================================

def normalize_name(x):

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


def name_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(r"\w+", str(x).casefold())
    )


# =========================================================
# 2. Load validation IDs
# =========================================================

val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids["source1_entity_id"]
)


# =========================================================
# 3. Load ground truth
# =========================================================

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


# =========================================================
# 4. Load validation Source 1
# =========================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1["norm_name"] = s1["business_name"].apply(
    normalize_name
)

s1["tokens"] = s1["business_name"].apply(
    name_tokens
)


# =========================================================
# 5. Load S2 + S3
# =========================================================

print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
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
        "country"
    ]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)


# =========================================================
# 6. Normalize target names
# =========================================================

print("Normalizing target names...")

targets["norm_name"] = targets[
    "business_name"
].apply(
    normalize_name
)

targets["tokens"] = targets[
    "business_name"
].apply(
    name_tokens
)


# =========================================================
# 7. Build exact-name index
# =========================================================

print("Building exact-name index...")

name_index = defaultdict(list)

for row in targets.itertuples(index=False):

    key = (
        row.country,
        row.norm_name
    )

    name_index[key].append(
        row.entity_id
    )


# =========================================================
# 8. Build token frequency
# =========================================================

print("Counting token frequencies...")

token_counts = Counter()

for tokens in targets["tokens"]:

    for token in tokens:
        token_counts[token] += 1


# =========================================================
# 9. Build country + token index
# =========================================================

print("Building token index...")

token_index = defaultdict(list)

for row in targets.itertuples(index=False):

    for token in row.tokens:

        token_index[
            (row.country, token)
        ].append(
            row.entity_id
        )


# =========================================================
# 10. Test thresholds
# =========================================================

thresholds = [
    100,
    500,
    1000
]

print("\n" + "=" * 85)
print("COMBINED BLOCKER RESULTS")
print("=" * 85)

print(
    f"{'Threshold':>12}"
    f"{'Candidates':>18}"
    f"{'Avg/S1':>15}"
    f"{'True Recovered':>18}"
    f"{'Recall':>15}"
)


for threshold in thresholds:

    total_candidates = 0
    recovered_true = 0
    total_true = 0

    for row in s1.itertuples(index=False):

        candidates = set()

        # -------------------------------------------------
        # BLOCK A: exact normalized name + country
        # -------------------------------------------------

        candidates.update(
            name_index.get(
                (row.country, row.norm_name),
                []
            )
        )

        # -------------------------------------------------
        # BLOCK B: rare token + country
        # -------------------------------------------------

        for token in row.tokens:

            if token_counts[token] <= threshold:

                candidates.update(
                    token_index.get(
                        (row.country, token),
                        []
                    )
                )

        total_candidates += len(candidates)

        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        true_string = gt_map.get(
            row.entity_id,
            ""
        )

        if true_string == "":
            true_matches = set()
        else:
            true_matches = set(
                true_string.split(",")
            )

        recovered = (
            candidates & true_matches
        )

        recovered_true += len(recovered)
        total_true += len(true_matches)

    recall = recovered_true / total_true

    print(
        f"{threshold:>12}"
        f"{total_candidates:>18}"
        f"{total_candidates / len(s1):>15.2f}"
        f"{recovered_true:>18}"
        f"{recall:>14.2%}"
    )