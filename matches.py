import pandas as pd

gt = pd.read_csv("train_ground_truth.tsv", sep="\t")

# Take 10 random Source 1 entities that have at least one match
sample = gt[gt["matched_entity_ids"].fillna("") != ""].sample(
    10, random_state=42
)

print(sample.to_string(index=False))




s1_id = "S1-793615008"

s1 = pd.read_csv("train_source1.tsv", sep="\t")
s2 = pd.read_csv("train_source2.tsv", sep="\t")
s3 = pd.read_csv("train_source3.tsv", sep="\t")

print("\nSOURCE 1")
print(s1[s1["entity_id"] == s1_id].to_string(index=False))

matched = gt.loc[
    gt["source1_entity_id"] == s1_id,
    "matched_entity_ids"
].iloc[0]

ids = matched.split(",")

print("\nMATCHED RECORDS")

print(
    s2[s2["entity_id"].isin(ids)].to_string(index=False)
)

print(
    s3[s3["entity_id"].isin(ids)].to_string(index=False)
)