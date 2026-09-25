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


def address_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(r"\w+", str(x).casefold())
    )


# =========================================================
# 2. Load validation IDs
# =========================================================

print("Loading validation IDs...")

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


# =========================================================
# 4. Load Source 1
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
].apply(
    normalize_name
)

s1["address_tokens"] = s1[
    "business_address"
].apply(
    address_tokens
)


# =========================================================
# 5. Load Source 2 + Source 3
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


# =========================================================
# 6. Normalize target data
# =========================================================

print("Normalizing target data...")

targets["norm_name"] = targets[
    "business_name"
].apply(
    normalize_name
)

targets["address_tokens"] = targets[
    "business_address"
].apply(
    address_tokens
)


# =========================================================
# 7. Exact-name index
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
# 8. Country-specific address token frequencies
# =========================================================

print("Counting address token frequencies...")

country_token_counts = {}

for country, group in targets.groupby("country"):

    counter = Counter()

    for tokens in group["address_tokens"]:

        for token in tokens:
            counter[token] += 1

    country_token_counts[country] = counter


# =========================================================
# 9. Country + address-token index
# =========================================================

print("Building address-token index...")

address_index = defaultdict(list)

for row in targets.itertuples(index=False):

    for token in row.address_tokens:

        address_index[
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


print("\n" + "=" * 90)
print("NAME + ADDRESS COMBINED BLOCKER")
print("=" * 90)

print(
    f"{'Threshold':>12}"
    f"{'S1 with candidates':>20}"
    f"{'Total candidates':>20}"
    f"{'Avg/S1':>15}"
    f"{'True recovered':>18}"
    f"{'Recall':>12}"
)


for threshold in thresholds:

    total_candidates = 0
    s1_with_candidates = 0

    recovered_true = 0
    total_true = 0

    for row in s1.itertuples(index=False):

        candidates = set()

        # -------------------------------------------------
        # BLOCK A: exact normalized name + country
        # -------------------------------------------------

        candidates.update(
            name_index.get(
                (
                    row.country,
                    row.norm_name
                ),
                []
            )
        )

        # -------------------------------------------------
        # BLOCK B: rare address token + country
        # -------------------------------------------------

        token_counts = country_token_counts.get(
            row.country,
            {}
        )

        for token in row.address_tokens:

            if token_counts.get(token, 0) <= threshold:

                candidates.update(
                    address_index.get(
                        (
                            row.country,
                            token
                        ),
                        []
                    )
                )

        # -------------------------------------------------
        # Candidate statistics
        # -------------------------------------------------

        if candidates:
            s1_with_candidates += 1

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

        recovered_true += len(
            candidates & true_matches
        )

        total_true += len(true_matches)

    recall = (
        recovered_true / total_true
        if total_true > 0
        else 0
    )

    print(
        f"{threshold:>12}"
        f"{s1_with_candidates:>20}"
        f"{total_candidates:>20}"
        f"{total_candidates / len(s1):>15.2f}"
        f"{recovered_true:>18}"
        f"{recall:>11.2%}"
    )