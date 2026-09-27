import pandas as pd

gt = pd.read_csv("train_ground_truth.tsv", sep="\t")

# Remove empty match lists
gt = gt[gt["matched_entity_ids"].fillna("") != ""].copy()

# Convert comma-separated IDs into individual rows
pairs = gt.assign(
    matched_entity_ids=gt["matched_entity_ids"].str.split(",")
).explode("matched_entity_ids")

# Count how many Source 1 entities each S2/S3 ID is associated with
counts = pairs["matched_entity_ids"].value_counts()

print("Total matched pairs:", len(pairs))
print("Unique matched S2/S3 IDs:", counts.size)

print("\nIDs appearing for MORE THAN ONE Source 1:")
print(counts[counts > 1].head(20))

print("\nNumber of S2/S3 IDs appearing multiple times:",
      (counts > 1).sum())

print("\nMaximum S1 entities linked to one S2/S3 ID:",
      counts.max())