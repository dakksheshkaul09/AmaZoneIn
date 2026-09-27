import pandas as pd

# Load ground truth file
df = pd.read_csv("train_ground_truth.tsv", sep="\t")

# Count number of matched entities for each Source 1 entity
df["match_count"] = df["matched_entity_ids"].fillna("").apply(
    lambda x: 0 if x.strip() == "" else len(x.split(","))
)

# 1. Number of Source 1 entities
print("1. Number of Source 1 entities:",
      df["source1_entity_id"].nunique())

# 2. How many have 0 matches?
print("2. Number with 0 matches:",
      (df["match_count"] == 0).sum())

# 3. How many have 1 match?
print("3. Number with 1 match:",
      (df["match_count"] == 1).sum())

# 4. How many have multiple matches?
print("4. Number with multiple matches:",
      (df["match_count"] > 1).sum())

# 5. Distribution of number of matches
print("\n5. Distribution of number of matches:")
print(df["match_count"].value_counts().sort_index())