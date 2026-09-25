import pandas as pd
import re
from collections import Counter, defaultdict


# =========================================================
# 1. Tokenization
# =========================================================

def address_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(r"\w+", str(x).casefold())
    )


# =========================================================
# 2. Load validation S1
# =========================================================

print("Loading validation IDs...")

val_ids = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids["source1_entity_id"]
)

print("Loading Source 1...")

s1 = pd.read_csv(
    "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_address",
        "country"
    ]
)

s1 = s1[
    s1["entity_id"].isin(val_ids)
].copy()

s1["tokens"] = s1[
    "business_address"
].apply(address_tokens)


# =========================================================
# 3. Load Source 2 + Source 3
# =========================================================

print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_address",
        "country"
    ]
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_address",
        "country"
    ]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

targets["tokens"] = targets[
    "business_address"
].apply(address_tokens)


# =========================================================
# 4. Country-specific token frequencies
# =========================================================

print("Counting address tokens...")

country_token_counts = {}

for country, group in targets.groupby("country"):

    counter = Counter()

    for tokens in group["tokens"]:

        for token in tokens:
            counter[token] += 1

    country_token_counts[country] = counter


# =========================================================
# 5. Build country + token index
# =========================================================

print("Building country-token index...")

index = defaultdict(list)

for row in targets.itertuples(index=False):

    for token in row.tokens:

        index[
            (row.country, token)
        ].append(
            row.entity_id
        )


# =========================================================
# 6. Test thresholds
# =========================================================

thresholds = [
    100,
    500,
    1000,
    5000,
    10000
]

print("\n" + "=" * 80)
print("ADDRESS TOKEN BLOCKING CANDIDATE COUNTS")
print("=" * 80)

print(
    f"{'Threshold':>12}"
    f"{'S1 with candidates':>22}"
    f"{'Total candidates':>20}"
    f"{'Avg candidates/S1':>22}"
)


for threshold in thresholds:

    total_candidates = 0
    s1_with_candidates = 0

    for row in s1.itertuples(index=False):

        candidates = set()

        for token in row.tokens:

            # Only use sufficiently rare tokens
            if country_token_counts[
                row.country
            ].get(token, 0) <= threshold:

                candidates.update(
                    index.get(
                        (row.country, token),
                        []
                    )
                )

        if candidates:
            s1_with_candidates += 1

        total_candidates += len(candidates)

    average = (
        total_candidates / len(s1)
    )

    print(
        f"{threshold:>12}"
        f"{s1_with_candidates:>22}"
        f"{total_candidates:>20}"
        f"{average:>22.2f}"
    )