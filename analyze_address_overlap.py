import pandas as pd
import re


# =========================================================
# 1. Address tokenization
# =========================================================

def address_tokens(x):

    if pd.isna(x):
        return set()

    x = str(x).casefold()

    return set(
        re.findall(r"\w+", x)
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

# Remove singletons
gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
].copy()


# =========================================================
# 4. Convert S1 → multiple matches into individual pairs
# =========================================================

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


# =========================================================
# 5. Load Source 1
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

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "s1_name",
        "business_address": "s1_address",
        "country": "s1_country"
    }
)


# =========================================================
# 6. Load Source 2
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
# 7. Load Source 3
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
# 8. Combine Source 2 + Source 3
# =========================================================

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

targets = targets.rename(
    columns={
        "entity_id": "target_id",
        "business_name": "target_name",
        "business_address": "target_address",
        "country": "target_country"
    }
)


# =========================================================
# 9. Join true pairs with actual records
# =========================================================

print("Joining true pairs...")

pairs = pairs.merge(
    s1[
        [
            "source1_entity_id",
            "s1_name",
            "s1_address",
            "s1_country"
        ]
    ],
    on="source1_entity_id",
    how="left"
)

pairs = pairs.merge(
    targets[
        [
            "target_id",
            "target_name",
            "target_address",
            "target_country"
        ]
    ],
    on="target_id",
    how="left"
)


# =========================================================
# 10. Create address tokens
# =========================================================

print("Calculating address token overlap...")

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
# 11. Number of shared address tokens
# =========================================================

pairs["shared_tokens"] = pairs.apply(
    lambda row:
        len(
            row["s1_tokens"] &
            row["target_tokens"]
        ),
    axis=1
)


# =========================================================
# 12. Jaccard similarity
# =========================================================

pairs["jaccard"] = pairs.apply(
    lambda row:

        len(
            row["s1_tokens"] &
            row["target_tokens"]
        )
        /
        len(
            row["s1_tokens"] |
            row["target_tokens"]
        )

        if len(
            row["s1_tokens"] |
            row["target_tokens"]
        ) > 0

        else 0,
    axis=1
)


# =========================================================
# 13. Print results
# =========================================================

print("\n" + "=" * 70)
print("ADDRESS TOKEN OVERLAP ON TRUE MATCHES")
print("=" * 70)

print(
    "Total true pairs:",
    len(pairs)
)

print(
    "At least 1 shared address token:",
    (pairs["shared_tokens"] >= 1).sum()
)

print(
    "At least 1 shared address token %:",
    (pairs["shared_tokens"] >= 1).mean()
)

print(
    "At least 2 shared address tokens:",
    (pairs["shared_tokens"] >= 2).sum()
)

print(
    "At least 2 shared address tokens %:",
    (pairs["shared_tokens"] >= 2).mean()
)

print(
    "Jaccard >= 0.25:",
    (pairs["jaccard"] >= 0.25).mean()
)

print(
    "Jaccard >= 0.50:",
    (pairs["jaccard"] >= 0.50).mean()
)

print(
    "Jaccard >= 0.75:",
    (pairs["jaccard"] >= 0.75).mean()
)