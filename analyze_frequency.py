import pandas as pd
import re
from collections import Counter


def name_tokens(x):
    if pd.isna(x):
        return set()

    x = str(x).casefold()

    return set(
        re.findall(r"\w+", x)
    )


print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=["business_name"]
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    usecols=["business_name"]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

print("Counting tokens...")

token_counts = Counter()

for name in targets["business_name"]:
    tokens = name_tokens(name)

    for token in tokens:
        token_counts[token] += 1


print("\n" + "=" * 70)
print("TOKEN FREQUENCY")
print("=" * 70)

print("Total unique tokens:", len(token_counts))

print("\nTop 30 most common tokens:")

for token, count in token_counts.most_common(30):
    print(
        f"{token:<30} {count:>10}"
    )