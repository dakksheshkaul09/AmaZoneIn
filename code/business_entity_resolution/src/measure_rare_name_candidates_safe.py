import pandas as pd
import re
import gc
import numpy as np
from collections import Counter, defaultdict


# =========================================================
# SETTINGS
# =========================================================

THRESHOLDS = [100, 500, 1000]
CHUNK_SIZE = 200_000


# =========================================================
# 1. NAME TOKENIZATION
# =========================================================

def name_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(
            r"\w+",
            str(x).casefold()
        )
    )


# =========================================================
# 2. LOAD VALIDATION S1
# =========================================================

print("Loading validation IDs...")

val_ids_df = pd.read_csv(
    "validation_s1_ids.tsv",
    sep="\t"
)

val_ids = set(
    val_ids_df["source1_entity_id"]
)

del val_ids_df


print("Loading validation Source 1...")

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
].reset_index(drop=True)

print(
    "Validation S1:",
    len(s1)
)


# =========================================================
# 3. BUILD VALIDATION TOKEN -> S1 INDEX
#
# key = (country, token)
# value = list of validation S1 row numbers
# =========================================================

print("Building validation token index...")

validation_token_index = defaultdict(list)

for s1_idx, row in enumerate(
    s1.itertuples(index=False)
):

    tokens = name_tokens(
        row.business_name
    )

    for token in tokens:

        validation_token_index[
            (
                row.country,
                token
            )
        ].append(
            s1_idx
        )


# =========================================================
# 4. FIRST PASS
# COUNT NAME TOKEN FREQUENCIES IN S2 + S3
# =========================================================

name_counts = defaultdict(Counter)


def count_tokens_in_file(path):

    print("\nCounting tokens in:", path)

    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(
            index=False
        ):

            country = row.country

            tokens = name_tokens(
                row.business_name
            )

            for token in tokens:

                name_counts[
                    country
                ][token] += 1

        processed += len(chunk)

        print(
            f"Processed {processed:,}"
        )

        del chunk

    gc.collect()


count_tokens_in_file(
    "train_source2.tsv"
)

count_tokens_in_file(
    "train_source3.tsv"
)

print("\nFirst pass complete.")


# =========================================================
# 5. PREPARE COUNTERS
#
# candidate_counts[t] =
# total unique target records that become
# candidates across all validation S1
# =========================================================

candidate_counts = {
    threshold: np.zeros(
        len(s1),
        dtype=np.int64
    )
    for threshold in THRESHOLDS
}


# Number of validation S1 entities receiving
# at least one candidate
has_candidate = {
    threshold: np.zeros(
        len(s1),
        dtype=np.bool_
    )
    for threshold in THRESHOLDS
}


# last_seen[t][s1_idx] stores the global target
# row number most recently counted for that S1.
#
# This prevents double-counting the same target
# when it shares multiple name tokens with an S1.
last_seen = {
    threshold: np.full(
        len(s1),
        -1,
        dtype=np.int64
    )
    for threshold in THRESHOLDS
}


# =========================================================
# 6. SECOND PASS
# COUNT UNIQUE CANDIDATES
# =========================================================

global_target_index = 0


def process_target_file(path):

    global global_target_index

    print("\nScanning:", path)

    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(
            index=False
        ):

            country = row.country

            tokens = name_tokens(
                row.business_name
            )

            for token in tokens:

                key = (
                    country,
                    token
                )

                # Is this token used by any
                # validation S1?
                validation_s1_indices = (
                    validation_token_index.get(
                        key
                    )
                )

                if not validation_s1_indices:
                    continue

                frequency = name_counts[
                    country
                ].get(
                    token,
                    0
                )

                if frequency == 0:
                    continue

                # -------------------------------------------------
                # Update every threshold for which this token
                # is considered rare.
                # -------------------------------------------------

                for threshold in THRESHOLDS:

                    if frequency > threshold:
                        continue

                    for s1_idx in validation_s1_indices:

                        # Only count this target ONCE
                        # for this S1.
                        if (
                            last_seen[threshold][s1_idx]
                            != global_target_index
                        ):

                            last_seen[
                                threshold
                            ][s1_idx] = (
                                global_target_index
                            )

                            candidate_counts[
                                threshold
                            ][s1_idx] += 1

                            has_candidate[
                                threshold
                            ][s1_idx] = True

            global_target_index += 1

        processed += len(chunk)

        print(
            f"Processed {processed:,}"
        )

        del chunk
        gc.collect()


process_target_file(
    "train_source2.tsv"
)

process_target_file(
    "train_source3.tsv"
)


# =========================================================
# 7. RESULTS
# =========================================================

print("\n" + "=" * 90)
print("RARE NAME TOKEN BLOCKING — CANDIDATE COUNTS")
print("=" * 90)

print(
    f"{'Threshold':>12}"
    f"{'S1 w/ candidates':>20}"
    f"{'Total candidates':>20}"
    f"{'Average/S1':>18}"
)

for threshold in THRESHOLDS:

    total = int(
        candidate_counts[
            threshold
        ].sum()
    )

    s1_count = int(
        has_candidate[
            threshold
        ].sum()
    )

    average = (
        total / len(s1)
    )

    print(
        f"{threshold:>12}"
        f"{s1_count:>20}"
        f"{total:>20}"
        f"{average:>18.2f}"
    )

print("\nDone.")