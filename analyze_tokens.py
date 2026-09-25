import pandas as pd
import re
from collections import Counter


def name_tokens(x):
    if pd.isna(x):
        return set()

    x = str(x).casefold()

    return set(
        re.findall(r"\w+", x)
    )


# -----------------------------------
# 1. Load all target names
# -----------------------------------

print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)


# -----------------------------------
# 2. Count token frequency
# -----------------------------------

print("Counting target token frequencies...")

token_counts = Counter()

for name in targets["business_name"]:
    for token in name_tokens(name):
        token_counts[token] += 1


# -----------------------------------
# 3. Load validation ground truth
# -----------------------------------

print("Loading ground truth...")

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t"
)

val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids["source1_entity_id"]
)

gt = gt[
    gt["source1_entity_id"].isin(val_ids)
].copy()

gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
]


# -----------------------------------
# 4. Convert to individual true pairs
# -----------------------------------

pairs = gt.assign(
    matched_entity_ids=gt["matched_entity_ids"].str.split(",")
).explode(
    "matched_entity_ids"
)

pairs = pairs.rename(
    columns={
        "matched_entity_ids": "target_id"
    }
)


# -----------------------------------
# 5. Add S1 and target names
# -----------------------------------

print("Loading Source 1...")

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "s1_name"
    }
)


pairs = pairs.merge(
    s1,
    on="source1_entity_id",
    how="left"
)

pairs = pairs.merge(
    targets.rename(
        columns={
            "entity_id": "target_id",
            "business_name": "target_name"
        }
    ),
    on="target_id",
    how="left"
)


# -----------------------------------
# 6. Calculate shared-token frequency
# -----------------------------------

print("Analyzing shared tokens...")

pairs["s1_tokens"] = pairs["s1_name"].apply(
    name_tokens
)

pairs["target_tokens"] = pairs["target_name"].apply(
    name_tokens
)

pairs["shared_tokens"] = pairs.apply(
    lambda row:
        row["s1_tokens"] & row["target_tokens"],
    axis=1
)

# For every true pair, find the RAREST
# shared token in the target corpus.
pairs["rarest_shared_frequency"] = pairs[
    "shared_tokens"
].apply(
    lambda tokens:
        min(
            [token_counts[t] for t in tokens],
            default=float("inf")
        )
)


# -----------------------------------
# 7. Test thresholds
# -----------------------------------

thresholds = [
    10,
    50,
    100,
    500,
    1000,
    5000,
    10000,
    50000
]

print("\n" + "=" * 75)
print("RARE TOKEN COVERAGE ON TRUE MATCHES")
print("=" * 75)

print(
    f"{'Threshold':>12} "
    f"{'Recovered':>15} "
    f"{'Coverage':>15}"
)

for threshold in thresholds:

    recovered = (
        pairs["rarest_shared_frequency"]
        <= threshold
    ).sum()

    coverage = recovered / len(pairs)

    print(
        f"{threshold:>12} "
        f"{recovered:>15} "
        f"{coverage:>14.2%}"
    )


# -----------------------------------
# 8. Distribution of rarest token
# -----------------------------------

print("\n" + "=" * 75)
print("RAREST SHARED TOKEN FREQUENCY")
print("=" * 75)

print(
    pairs["rarest_shared_frequency"].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)