import pandas as pd
import re
import unicodedata


def normalize_name(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()

    x = unicodedata.normalize("NFKD", x)
    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )

    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = " ".join(x.split())

    return x


# -------------------------
# Load validation S1
# -------------------------

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t"
)

val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

val_s1 = s1[
    s1["entity_id"].isin(
        val_ids["source1_entity_id"]
    )
].copy()

# -------------------------
# Load S2 + S3
# -------------------------

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

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

# -------------------------
# Normalize names
# -------------------------

val_s1["norm_name"] = val_s1["business_name"].apply(
    normalize_name
)

targets["norm_name"] = targets["business_name"].apply(
    normalize_name
)

# -------------------------
# Build blocker index
# key = country + normalized name
# -------------------------

index = {}

for row in targets.itertuples(index=False):

    key = (
        row.country,
        row.norm_name
    )

    if key not in index:
        index[key] = []

    index[key].append(row.entity_id)

# -------------------------
# Calculate candidate count
# -------------------------

total_candidates = 0
s1_with_candidates = 0

for row in val_s1.itertuples(index=False):

    key = (
        row.country,
        row.norm_name
    )

    candidates = index.get(key, [])

    total_candidates += len(candidates)

    if candidates:
        s1_with_candidates += 1

print("Validation S1:", len(val_s1))
print("S1 with >=1 candidate:", s1_with_candidates)
print("Total candidates:", total_candidates)

print(
    "Average candidates per S1:",
    total_candidates / len(val_s1)
)