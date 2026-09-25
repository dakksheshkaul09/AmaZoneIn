import pandas as pd

df = pd.read_csv("train_source2.tsv", sep="\t")

dupes = df[df["entity_id"].duplicated(keep=False)].sort_values("entity_id")

print(dupes)

print("Duplicate entity IDs:", df["entity_id"].duplicated().sum())