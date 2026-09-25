import pandas as pd
import re


def name_tokens(x):
    if pd.isna(x):
        return set()

    x = str(x).casefold()

    # Unicode-aware word extraction
    tokens = re.findall(r"\w+", x)

    return set(tokens)


# -----------------------------------
# Load validation IDs
# -----------------------------------

val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids["source1_entity_id"]
)


# -----------------------------------
# Load ground truth
# -----------------------------------

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(val_ids)
]

gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
]

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
# Load S1
# -----------------------------------

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
]

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "s1_name"
    }
)


# -----------------------------------
# Load S2 + S3
# -----------------------------------

cols = [
    "entity_id",
    "business_name"
]

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=cols
)

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    usecols=cols
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

targets = targets.rename(
    columns={
        "entity_id": "target_id",
        "business_name": "target_name"
    }
)


# -----------------------------------
# Join true pairs
# -----------------------------------

pairs = pairs.merge(
    s1,
    on="source1_entity_id",
    how="left"
)

pairs = pairs.merge(
    targets,
    on="target_id",
    how="left"
)


# -----------------------------------
# Calculate token overlap
# -----------------------------------

pairs["s1_tokens"] = pairs["s1_name"].apply(
    name_tokens
)

pairs["target_tokens"] = pairs["target_name"].apply(
    name_tokens
)

pairs["shared_tokens"] = pairs.apply(
    lambda x: len(
        x["s1_tokens"] &
        x["target_tokens"]
    ),
    axis=1
)

pairs["jaccard"] = pairs.apply(
    lambda x:
        len(x["s1_tokens"] & x["target_tokens"])
        /
        len(x["s1_tokens"] | x["target_tokens"])
        if len(x["s1_tokens"] | x["target_tokens"]) > 0
        else 0,
    axis=1
)


# -----------------------------------
# Results
# -----------------------------------

print("\n" + "=" * 70)
print("NAME TOKEN OVERLAP ON TRUE MATCHES")
print("=" * 70)

print(
    "Total true pairs:",
    len(pairs)
)

print(
    "At least 1 shared token:",
    (pairs["shared_tokens"] >= 1).sum()
)

print(
    "At least 2 shared tokens:",
    (pairs["shared_tokens"] >= 2).sum()
)

print(
    "At least 1 shared token %:",
    (pairs["shared_tokens"] >= 1).mean()
)

print(
    "At least 2 shared tokens %:",
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