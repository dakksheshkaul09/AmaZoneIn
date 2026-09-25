import pandas as pd
import re
from collections import Counter, defaultdict


def name_tokens(x):
    if pd.isna(x):
        return set()

    return set(
        re.findall(r"\w+", str(x).casefold())
    )


# -----------------------------------
# Load validation S1
# -----------------------------------

val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids["source1_entity_id"]
)

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

s1["tokens"] = s1["business_name"].apply(
    name_tokens
)


# -----------------------------------
# Load S2 + S3
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


# -----------------------------------
# Build token frequencies
# -----------------------------------

print("Counting tokens...")

token_counts = Counter()

for name in targets["business_name"]:
    for token in name_tokens(name):
        token_counts[token] += 1


# -----------------------------------
# Build country + token index
# -----------------------------------

print("Building country-token index...")

index = defaultdict(list)

for row in targets.itertuples(index=False):

    tokens = name_tokens(row.business_name)

    for token in tokens:
        index[
            (row.country, token)
        ].append(row.entity_id)


# -----------------------------------
# Test thresholds
# -----------------------------------

thresholds = [
    100,
    500,
    1000,
    5000,
    10000,
    50000
]


print("\n" + "=" * 75)
print("RARE TOKEN BLOCKING CANDIDATE COUNTS")
print("=" * 75)

print(
    f"{'Threshold':>12}"
    f"{'S1 with candidates':>20}"
    f"{'Total candidates':>20}"
    f"{'Avg candidates/S1':>20}"
)


for threshold in thresholds:

    total_candidates = 0
    s1_with_candidates = 0

    for row in s1.itertuples(index=False):

        candidate_ids = set()

        for token in row.tokens:

            if token_counts[token] <= threshold:

                candidate_ids.update(
                    index.get(
                        (row.country, token),
                        []
                    )
                )

        if candidate_ids:
            s1_with_candidates += 1

        total_candidates += len(candidate_ids)

    average = (
        total_candidates / len(s1)
    )

    print(
        f"{threshold:>12}"
        f"{s1_with_candidates:>20}"
        f"{total_candidates:>20}"
        f"{average:>20.2f}"
    )