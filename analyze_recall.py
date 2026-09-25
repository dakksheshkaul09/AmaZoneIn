import pandas as pd
import unicodedata


# -----------------------------------
# 1. Name normalization
# -----------------------------------

def normalize_name(x):
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
    x = " ".join(x.split())

    return x


# -----------------------------------
# 2. Load ground truth
# -----------------------------------

print("Loading ground truth...")

gt = pd.read_csv(
    "train_ground_truth.tsv",
    sep="\t"
)

# Validation S1 IDs
val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)


# -----------------------------------
# 3. Load Source 1
# -----------------------------------

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

# Keep ONLY validation S1 entities
val_s1 = s1[
    s1["entity_id"].isin(
        val_ids["source1_entity_id"]
    )
].copy()

val_s1["norm_name"] = val_s1["business_name"].apply(
    normalize_name
)

print("Validation S1:", len(val_s1))


# -----------------------------------
# 4. Load Source 2
# -----------------------------------

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

s2["norm_name"] = s2["business_name"].apply(
    normalize_name
)


# -----------------------------------
# 5. Load Source 3
# -----------------------------------

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

s3["norm_name"] = s3["business_name"].apply(
    normalize_name
)


# -----------------------------------
# 6. Combine S2 + S3
# -----------------------------------

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)


# -----------------------------------
# 7. Build blocking index
#
# key = (country, normalized name)
# -----------------------------------

print("Building blocking index...")

index = {}

for row in targets.itertuples(index=False):

    key = (
        row.country,
        row.norm_name
    )

    if key not in index:
        index[key] = []

    index[key].append(row.entity_id)


# -----------------------------------
# 8. Ground truth lookup
# -----------------------------------

gt_val = gt[
    gt["source1_entity_id"].isin(
        val_ids["source1_entity_id"]
    )
].copy()

gt_map = dict(
    zip(
        gt_val["source1_entity_id"],
        gt_val["matched_entity_ids"].fillna("")
    )
)


from collections import defaultdict


# -----------------------------------
# Calculate recall by match count
# -----------------------------------

results = defaultdict(
    lambda: {
        "true_matches": 0,
        "recovered_matches": 0,
        "entities": 0
    }
)


for row in val_s1.itertuples(index=False):

    s1_id = row.entity_id

    key = (
        row.country,
        row.norm_name
    )

    candidates = set(
        index.get(key, [])
    )

    true_string = gt_map.get(
        s1_id,
        ""
    )

    if true_string == "":
        true_matches = set()
    else:
        true_matches = set(
            true_string.split(",")
        )

    recovered = candidates & true_matches

    match_count = len(true_matches)

    results[match_count]["entities"] += 1
    results[match_count]["true_matches"] += len(true_matches)
    results[match_count]["recovered_matches"] += len(recovered)


# -----------------------------------
# Print results
# -----------------------------------

print("\n" + "=" * 70)
print("BLOCKING RECALL BY TRUE MATCH COUNT")
print("=" * 70)

print(
    f"{'Matches':>8} "
    f"{'Entities':>12} "
    f"{'True':>12} "
    f"{'Recovered':>12} "
    f"{'Recall':>12}"
)

for match_count in sorted(results):

    true_count = results[match_count]["true_matches"]
    recovered_count = results[match_count]["recovered_matches"]
    entities = results[match_count]["entities"]

    if true_count > 0:
        recall = recovered_count / true_count
    else:
        recall = 1.0

    print(
        f"{match_count:>8} "
        f"{entities:>12} "
        f"{true_count:>12} "
        f"{recovered_count:>12} "
        f"{recall:>11.2%}"
    )