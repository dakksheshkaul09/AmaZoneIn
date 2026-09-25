import pandas as pd
from validation import macro_f05

gt = pd.read_csv("train_ground_truth.tsv", sep="\t")

val_ids = pd.read_csv("validation_s1_ids.tsv", sep="\t")

val_gt = gt[
    gt["source1_entity_id"].isin(
        val_ids["source1_entity_id"]
    )
].copy()

score = macro_f05(val_gt, val_gt)

print("Validation rows:", len(val_gt))
print("Perfect prediction F0.5:", score)