import pandas as pd
from sklearn.model_selection import train_test_split

gt = pd.read_csv("train_ground_truth.tsv", sep="\t")

# Number of true matches for each S1
gt["match_count"] = gt["matched_entity_ids"].fillna("").apply(
    lambda x: 0 if x == "" else len(x.split(","))
)

# Stratify by both country and match-count bucket
s1 = pd.read_csv("train_source1.tsv", sep="\t")

gt = gt.merge(
    s1[["entity_id", "country"]],
    left_on="source1_entity_id",
    right_on="entity_id",
    how="left"
)

gt["strata"] = (
    gt["country"].astype(str)
    + "_"
    + gt["match_count"].astype(str)
)

train_gt, val_gt = train_test_split(
    gt,
    test_size=0.05,
    random_state=42,
    stratify=gt["strata"]
)

print("Training S1:", len(train_gt))
print("Validation S1:", len(val_gt))

print("\nTraining match-count distribution:")
print(train_gt["match_count"].value_counts(normalize=True).sort_index())

print("\nValidation match-count distribution:")
print(val_gt["match_count"].value_counts(normalize=True).sort_index())

# Save ONLY the S1 IDs for the validation split
val_gt[["source1_entity_id"]].to_csv(
    "validation_s1_ids.tsv",
    sep="\t",
    index=False
)