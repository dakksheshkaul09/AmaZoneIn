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


# -----------------------------------
# 9. Calculate blocking recall
# -----------------------------------

print("Calculating blocking recall...")

total_true_matches = 0
recovered_true_matches = 0

all_recovered = 0
some_recovered = 0
zero_recovered = 0

total_candidates = 0


for row in val_s1.itertuples(index=False):

    s1_id = row.entity_id

    # Candidate lookup
    key = (
        row.country,
        row.norm_name
    )

    candidates = set(
        index.get(key, [])
    )

    total_candidates += len(candidates)

    # Ground truth
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

    # True matches recovered by blocker
    recovered = candidates & true_matches

    total_true_matches += len(true_matches)
    recovered_true_matches += len(recovered)

    # Entity-level recovery
    if len(recovered) == len(true_matches):
        all_recovered += 1
    elif len(recovered) > 0:
        some_recovered += 1
    else:
        zero_recovered += 1


# -----------------------------------
# 10. Print results
# -----------------------------------

print("\n" + "=" * 50)
print("BLOCKING BASELINE RESULTS")
print("=" * 50)

print(
    "Validation S1:",
    len(val_s1)
)

print(
    "Total candidates:",
    total_candidates
)

print(
    "Average candidates per S1:",
    total_candidates / len(val_s1)
)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "Recovered true matches:",
    recovered_true_matches
)

print(
    "Candidate recall:",
    recovered_true_matches / total_true_matches
)

print(
    "ALL true matches recovered:",
    all_recovered
)

print(
    "SOME true matches recovered:",
    some_recovered
)

print(
    "ZERO true matches recovered:",
    zero_recovered
)