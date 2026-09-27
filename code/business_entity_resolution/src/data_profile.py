import pandas as pd

# Enter the name of ONE file
file_name = input("Enter TSV file name: ")

# Read the file
df = pd.read_csv(file_name, sep="\t")

print("=" * 60)
print("FILE:", file_name)
print("=" * 60)

# 1. Number of rows
print("\n1. Number of rows:", df.shape[0])

# 2. Number of columns
print("2. Number of columns:", df.shape[1])

# 3. Column names
print("\n3. Column names:")
print(list(df.columns))

# 4. Missing values per column
print("\n4. Missing values per column:")
print(df.isnull().sum())

# 5. Number of unique entity_ids
print("\n5. Number of unique entity_ids:")
print(df["entity_id"].nunique())

# 6. Country counts
print("\n6. Country counts:")
print(df["country"].value_counts(dropna=False))

# 7. 10 random/sample rows
print("\n7. 10 random/sample rows:")
print(df.sample(min(10, len(df)), random_state=42))