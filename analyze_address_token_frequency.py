import pandas as pd
import re
from collections import Counter


def address_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(r"\w+", str(x).casefold())
    )


print("Loading Source 2...")

s2 = pd.read_csv(
    "train_source2.tsv",
    sep="\t",
    usecols=[
        "business_address",
        "country"
    ]
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "train_source3.tsv",
    sep="\t",
    usecols=[
        "business_address",
        "country"
    ]
)

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)


# -------------------------------------------------
# Count address-token frequency WITHIN each country
# -------------------------------------------------

print("Counting country-specific address tokens...")

country_token_counts = {}

for country, group in targets.groupby("country"):

    counter = Counter()

    for address in group["business_address"]:

        for token in address_tokens(address):
            counter[token] += 1

    country_token_counts[country] = counter


# -------------------------------------------------
# Print results
# -------------------------------------------------

print("\n" + "=" * 75)
print("ADDRESS TOKEN FREQUENCY BY COUNTRY")
print("=" * 75)

for country in sorted(country_token_counts):

    print("\n" + "-" * 50)
    print("COUNTRY:", country)
    print("-" * 50)

    counter = country_token_counts[country]

    print("Unique address tokens:", len(counter))

    print("\nTop 30 most common tokens:")

    for token, count in counter.most_common(30):

        print(
            f"{token:<30} {count:>10}"
        )