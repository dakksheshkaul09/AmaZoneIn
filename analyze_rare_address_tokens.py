import pandas as pd
import re
from collections import Counter


# =========================================================
# 1. Address tokenization
# =========================================================

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
# 3. Load Source 2 + Source 3
# =========================================================

print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
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
        "business_address",
        "country"
    ]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)


# =========================================================
# 4. Country-specific address token frequencies
# =========================================================

print("Counting address token frequencies...")

country_token_counts = {}

for country, group in targets.groupby("country"):

    counter = Counter()

    for address in group["business_address"]:

        for token in address_tokens(address):
            counter[token] += 1

    country_token_counts[country] = counter


# =========================================================
# 5. Load ground truth
# =========================================================

print("Loading ground truth...")

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(val_ids)
].copy()

# Remove singletons
gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
].copy()


# =========================================================
# 6. Convert S1 -> target list into individual pairs
# =========================================================

pairs = gt.assign(
    matched_entity_ids=gt[
        "matched_entity_ids"
    ].str.split(",")
).explode(
    "matched_entity_ids"
)

pairs = pairs.rename(
    columns={
        "matched_entity_ids": "target_id"
    }
)


# =========================================================
# 7. Load Source 1
# =========================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_address",
        "country"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_address": "s1_address",
        "country": "s1_country"
    }
)


# =========================================================
# 8. Join true pairs with S1 records
# =========================================================

pairs = pairs.merge(
    s1[
        [
            "source1_entity_id",
            "s1_address",
            "s1_country"
        ]
    ],
    on="source1_entity_id",
    how="left"
)


# =========================================================
# 9. Join true pairs with S2/S3 records
# =========================================================

targets_small = targets.rename(
    columns={
        "entity_id": "target_id",
        "business_address": "target_address",
        "country": "target_country"
    }
)

pairs = pairs.merge(
    targets_small[
        [
            "target_id",
            "target_address",
            "target_country"
        ]
    ],
    on="target_id",
    how="left"
)


# =========================================================
# 10. Tokenize addresses
# =========================================================

print("Analyzing shared address tokens...")

pairs["s1_tokens"] = pairs[
    "s1_address"
].apply(
    address_tokens
)

pairs["target_tokens"] = pairs[
    "target_address"
].apply(
    address_tokens
)


# =========================================================
# 11. Find rarest shared token
# =========================================================

def rarest_shared_frequency(row):

    shared = (
        row["s1_tokens"]
        &
        row["target_tokens"]
    )

    if not shared:
        return float("inf")

    counter = country_token_counts.get(
        row["s1_country"],
        {}
    )

    frequencies = [
        counter.get(token, 0)
        for token in shared
    ]

    return min(frequencies)


pairs["rarest_shared_frequency"] = pairs.apply(
    rarest_shared_frequency,
    axis=1
)


# =========================================================
# 12. Test thresholds
# =========================================================

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
print("RARE ADDRESS TOKEN COVERAGE ON TRUE MATCHES")
print("=" * 75)

print(
    f"{'Threshold':>12}"
    f"{'Recovered':>15}"
    f"{'Coverage':>15}"
)

for threshold in thresholds:

    recovered = (
        pairs["rarest_shared_frequency"]
        <= threshold
    ).sum()

    coverage = recovered / len(pairs)

    print(
        f"{threshold:>12}"
        f"{recovered:>15}"
        f"{coverage:>14.2%}"
    )


# =========================================================
# 13. Distribution
# =========================================================

print("\n" + "=" * 75)
print("RAREST SHARED ADDRESS TOKEN FREQUENCY")
print("=" * 75)

finite = pairs.loc[
    pairs["rarest_shared_frequency"] != float("inf"),
    "rarest_shared_frequency"
]

print(
    finite.describe(
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