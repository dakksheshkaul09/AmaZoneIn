import pandas as pd
import unicodedata


def normalize_text(x):

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

gt = gt[
    gt["matched_entity_ids"].fillna("") != ""
].copy()


# Convert each S1 → multiple targets
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
# Load Source 1
# -----------------------------------

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

s1["s1_norm_address"] = s1["s1_address"].apply(
    normalize_text
)


# -----------------------------------
# Load S2 + S3
# -----------------------------------

cols = [
    "entity_id",
    "business_name",
    "business_address",
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

targets = targets.rename(
    columns={
        "entity_id": "target_id",
        "business_name": "target_name",
        "business_address": "target_address",
        "country": "target_country"
    }
)

targets["target_norm_address"] = targets[
    "target_address"
].apply(normalize_text)


# -----------------------------------
# Join true pairs with actual records
# -----------------------------------

pairs = pairs.merge(
    s1[
        [
            "source1_entity_id",
            "s1_name",
            "s1_address",
            "s1_country",
            "s1_norm_address"
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
            "target_country",
            "target_norm_address"
        ]
    ],
    on="target_id",
    how="left"
)


# -----------------------------------
# Compare address
# -----------------------------------

pairs["address_same"] = (
    pairs["s1_norm_address"] ==
    pairs["target_norm_address"]
)

pairs["name_same"] = False

print("\n" + "=" * 70)
print("ADDRESS SIGNAL ON TRUE MATCHES")
print("=" * 70)

print(
    "Total true pairs:",
    len(pairs)
)

print(
    "Same normalized address:",
    pairs["address_same"].sum()
)

print(
    "Address match rate:",
    pairs["address_same"].mean()
)


# -----------------------------------
# Specifically inspect the
# name-different matches
# -----------------------------------

name_diff = pairs[
    ~pairs["name_same"]
].copy()

print("\n" + "=" * 70)
print("ADDRESS SIGNAL")
print("=" * 70)

print(
    "True pairs with same normalized address:",
    name_diff["address_same"].sum()
)

print(
    "Percentage:",
    name_diff["address_same"].mean()
)


# -----------------------------------
# Show examples
# -----------------------------------

print("\n" + "=" * 70)
print("EXAMPLES")
print("=" * 70)

print(
    name_diff[
        name_diff["address_same"]
    ][
        [
            "s1_name",
            "target_name",
            "s1_address",
            "target_address"
        ]
    ].head(15).to_string(index=False)
)