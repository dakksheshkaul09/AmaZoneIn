import pandas as pd
import unicodedata


# -----------------------------------
# Name normalization
# -----------------------------------

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

    x = " ".join(x.split())

    return x


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
].copy()

# Remove singleton rows
gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
].copy()

# Split true matches into individual pairs
pairs = gt[
    [
        "source1_entity_id",
        "matched_entity_ids"
    ]
].assign(
    matched_entity_ids=lambda x:
        x["matched_entity_ids"].str.split(",")
).explode(
    "matched_entity_ids"
)

pairs = pairs.rename(
    columns={
        "matched_entity_ids": "target_id"
    }
)

print("True validation pairs:", len(pairs))


# -----------------------------------
# Load Source 1
# -----------------------------------

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

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "s1_name",
        "country": "s1_country",
        "norm_name": "s1_norm_name"
    }
)


# -----------------------------------
# Load Source 2 + Source 3
# -----------------------------------

cols = [
    "entity_id",
    "business_name",
    "country"
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

targets["norm_name"] = targets["business_name"].apply(
    normalize_name
)

targets = targets.rename(
    columns={
        "entity_id": "target_id",
        "business_name": "target_name",
        "country": "target_country",
        "norm_name": "target_norm_name"
    }
)


# -----------------------------------
# Join S1 → true target records
# -----------------------------------

pairs = pairs.merge(
    s1[
        [
            "source1_entity_id",
            "s1_name",
            "s1_country",
            "s1_norm_name"
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
            "target_country",
            "target_norm_name"
        ]
    ],
    on="target_id",
    how="left"
)


# -----------------------------------
# Compare blocking conditions
# -----------------------------------

pairs["country_same"] = (
    pairs["s1_country"] ==
    pairs["target_country"]
)

pairs["name_same"] = (
    pairs["s1_norm_name"] ==
    pairs["target_norm_name"]
)


# -----------------------------------
# Categorize missed true matches
# -----------------------------------

pairs["category"] = "BOTH_DIFFER"

pairs.loc[
    pairs["country_same"] &
    ~pairs["name_same"],
    "category"
] = "COUNTRY_SAME_NAME_DIFFERENT"

pairs.loc[
    ~pairs["country_same"] &
    pairs["name_same"],
    "category"
] = "COUNTRY_DIFFERENT_NAME_SAME"

pairs.loc[
    pairs["country_same"] &
    pairs["name_same"],
    "category"
] = "BOTH_SAME"


# -----------------------------------
# Results
# -----------------------------------

print("\n" + "=" * 70)
print("TRUE MATCH ANALYSIS")
print("=" * 70)

print(
    pairs["category"].value_counts()
)

print("\nPercentages:")

print(
    pairs["category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------------
# Show examples
# -----------------------------------

print("\n" + "=" * 70)
print("EXAMPLES OF COUNTRY-SAME / NAME-DIFFERENT")
print("=" * 70)

print(
    pairs[
        pairs["category"] ==
        "COUNTRY_SAME_NAME_DIFFERENT"
    ][
        [
            "s1_name",
            "target_name",
            "s1_country",
            "target_country"
        ]
    ].head(15).to_string(index=False)
)