
import argparse
import gc
import heapq
import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# =========================================================
# V4 MEMORY-SAFE
# =========================================================
# Same V4 candidate stages and same matcher as the 0.706265
# validation experiment, but designed for a 16 GB laptop.
#
# Main memory fixes:
#   * DO NOT store S1 token/3-gram sets for all 1.7M rows.
#   * Blocking dictionaries store S1 integer indices.
#   * S1 keeps only normalized name/address strings.
#   * TOP-50 entries are packed into one Python integer:
#         score(0..1000) + target_index
#   * Target IDs are stored in a compact numpy memmap on disk.
#
# Run validation FIRST:
#   python v4_memory_safe.py --mode validation
#
# Then full test:
#   python v4_memory_safe.py --mode test
# =========================================================


CHUNK_SIZE = 50_000
GRAM_MAX_FREQ = 5000
RARE_FREQ = 100
TOP_K = 50
THRESHOLD = 0.86
NAME_WEIGHT = 0.40
ADDRESS_WEIGHT = 0.60

# Target index uses 24 bits: enough for the ~10M S2+S3 test rows.
TARGET_BITS = 24
TARGET_MASK = (1 << TARGET_BITS) - 1

# IDs are stored as fixed-width bytes in the memmap.
TARGET_ID_BYTES = 24


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).casefold()
    x = unicodedata.normalize("NFKD", x)

    x = "".join(
        c for c in x
        if not unicodedata.combining(c)
    )

    x = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in x
    )

    return " ".join(x.split())


def tokens(x):
    if not x:
        return set()

    return set(re.findall(r"\w+", x))


# =========================================================
# STRONG NAME
# =========================================================

LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "private": "private",
    "ltd": "limited",
    "limited": "limited",
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "lp": "lp",
}

LEGAL_SUFFIXES = {
    "private",
    "limited",
    "corporation",
    "incorporated",
    "company",
    "llc",
    "llp",
    "plc",
    "lp",
}


def strong_name(x):
    x = normalize_text(x)

    if not x:
        return ""

    return " ".join(
        LEGAL_SUFFIX_MAP.get(t, t)
        for t in x.split()
    )


def strong_name_core(x):
    x = strong_name(x)

    if not x:
        return ""

    ts = set(x.split())
    ts -= LEGAL_SUFFIXES

    return " ".join(sorted(ts))


# =========================================================
# STRONG ADDRESS
# =========================================================

ADDRESS_MAP = {
    "rd": "road",
    "road": "road",
    "st": "street",
    "street": "street",
    "ave": "avenue",
    "av": "avenue",
    "avenue": "avenue",
    "dr": "drive",
    "drive": "drive",
    "ln": "lane",
    "lane": "lane",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "hwy": "highway",
    "highway": "highway",
    "pkwy": "parkway",
    "parkway": "parkway",
    "ct": "court",
    "court": "court",
    "cir": "circle",
    "circle": "circle",
    "pl": "place",
    "place": "place",
    "ter": "terrace",
    "terrace": "terrace",
    "trl": "trail",
    "trail": "trail",
    "apt": "apartment",
    "appt": "apartment",
    "apartment": "apartment",
    "fl": "floor",
    "floor": "floor",
    "rm": "room",
    "room": "room",
    "ste": "suite",
    "suite": "suite",
    "bldg": "building",
    "building": "building",
    "no": "number",
    "num": "number",
    "number": "number",
}


def strong_address(x):
    x = normalize_text(x)

    if not x:
        return ""

    return " ".join(
        sorted(
            {
                ADDRESS_MAP.get(t, t)
                for t in x.split()
            }
        )
    )


# =========================================================
# CHARACTER 3-GRAMS
# =========================================================

def char3grams(x):
    if not x:
        return set()

    compact = "".join(
        c for c in x
        if c.isalnum()
    )

    if len(compact) < 3:
        return set()

    return {
        compact[i:i + 3]
        for i in range(len(compact) - 2)
    }


# =========================================================
# COMPACT LOOKUP
# =========================================================
# Value is either:
#   int          -> one S1 row
#   list[int]    -> multiple S1 rows
#
# This saves a lot of RAM compared with list-for-every-key.


def lookup_add(d, key, idx):
    old = d.get(key)

    if old is None:
        d[key] = idx

    elif isinstance(old, int):
        d[key] = [old, idx]

    else:
        old.append(idx)


def lookup_iter(value):
    if value is None:
        return ()

    if isinstance(value, int):
        return (value,)

    return value


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["validation", "test"],
        default="validation",
    )

    p.add_argument(
        "--train-dir",
        default="dataset/train",
    )

    p.add_argument(
        "--test-dir",
        default="dataset/test",
    )

    p.add_argument(
        "--validation-ids",
        default="validation_s1_ids.tsv",
    )

    p.add_argument(
        "--output-dir",
        default="output/v4_safe",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    p.add_argument(
        "--gram-max-freq",
        type=int,
        default=GRAM_MAX_FREQ,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
    )

    p.add_argument(
        "--limit-s1",
        type=int,
        default=None,
        help="Test-mode smoke test: load only the first N S1 rows.",
    )

    p.add_argument(
        "--limit-target-rows",
        type=int,
        default=None,
        help="Test-mode smoke test: read only the first N rows from each target file.",
    )

    return p.parse_args()


# =========================================================
# TARGET FREQUENCIES
# =========================================================

def count_target_features(paths, chunk_size, gram_max_freq):
    name_counts = {
        "US": Counter(),
        "India": Counter(),
        "France": Counter(),
    }

    address_counts = {
        "US": Counter(),
        "India": Counter(),
        "France": Counter(),
    }

    name_gram_counts = {
        "US": Counter(),
        "India": Counter(),
        "France": Counter(),
    }

    address_gram_counts = {
        "US": Counter(),
        "India": Counter(),
        "France": Counter(),
    }

    for path in paths:
        print("\nCounting:", path)

        processed = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=[
                "business_name",
                "business_address",
                "country",
            ],
            chunksize=chunk_size,
        ):
            for row in chunk.itertuples(index=False):
                c = row.country

                n = normalize_text(row.business_name)
                a = normalize_text(row.business_address)

                for t in tokens(n):
                    name_counts[c][t] += 1

                for t in tokens(a):
                    address_counts[c][t] += 1

                cn = strong_name_core(row.business_name)
                sa = strong_address(row.business_address)

                for g in char3grams(cn):
                    name_gram_counts[c][g] += 1

                for g in char3grams(sa):
                    address_gram_counts[c][g] += 1

            processed += len(chunk)
            print(
                f"  Processed {processed:,}",
                flush=True,
            )

            del chunk

        gc.collect()

    return (
        name_counts,
        address_counts,
        name_gram_counts,
        address_gram_counts,
    )


# =========================================================
# LOAD S1
# =========================================================

def load_s1(path, validation_ids=None, limit=None):
    print("\nLoading Source 1...")

    read_kwargs = dict(
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )
    if limit is not None:
        read_kwargs["nrows"] = limit

    df = pd.read_csv(
        path,
        **read_kwargs,
    )

    if validation_ids is not None:
        df = df[
            df["entity_id"].isin(validation_ids)
        ].copy()

    s1_ids = []
    s1_names = []
    s1_addresses = []

    # Build only the strings actually needed by the matcher.
    # No token sets / gram sets are retained.
    for row in df.itertuples(index=False):
        s1_ids.append(row.entity_id)
        s1_names.append(
            normalize_text(row.business_name)
        )
        s1_addresses.append(
            normalize_text(row.business_address)
        )

    del df
    gc.collect()

    print("S1 rows:", f"{len(s1_ids):,}")

    return (
        s1_ids,
        s1_names,
        s1_addresses,
    )


# =========================================================
# BUILD ALL 7 BLOCKING LOOKUPS
# =========================================================

def build_lookups(
    s1_path,
    validation_ids,
    name_counts,
    address_counts,
    name_gram_counts,
    address_gram_counts,
    gram_max_freq,
    s1_limit=None,
):
    print("\nBuilding compact blocking lookups...")

    exact_name = {}
    core_name = {}
    strong_addr = {}
    rare_name = {}
    rare_addr = {}
    name_gram = {}
    addr_gram = {}

    read_kwargs = dict(
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    if s1_limit is not None:
        read_kwargs["nrows"] = s1_limit

    for df in pd.read_csv(
        s1_path,
        chunksize=50_000,
        **read_kwargs,
    ):
        for row in df.itertuples(index=False):
            sid = row.entity_id

            if (
                validation_ids is not None
                and sid not in validation_ids
            ):
                continue

            # The S1 index must equal the order used by load_s1().
            #
            # We maintain a separate counter rather than parsing IDs.
            idx = build_lookups.current_idx
            build_lookups.current_idx += 1

            c = row.country

            n = normalize_text(
                row.business_name
            )

            a = normalize_text(
                row.business_address
            )

            cn = strong_name_core(
                row.business_name
            )

            sa = strong_address(
                row.business_address
            )

            if n:
                lookup_add(
                    exact_name,
                    (c, n),
                    idx,
                )

            if cn:
                lookup_add(
                    core_name,
                    (c, cn),
                    idx,
                )

            if sa:
                lookup_add(
                    strong_addr,
                    (c, sa),
                    idx,
                )

            nc = name_counts[c]

            for t in tokens(n):
                f = nc.get(t, 0)

                if 0 < f <= RARE_FREQ:
                    lookup_add(
                        rare_name,
                        (c, t),
                        idx,
                    )

            ac = address_counts[c]

            for t in tokens(a):
                f = ac.get(t, 0)

                if 0 < f <= RARE_FREQ:
                    lookup_add(
                        rare_addr,
                        (c, t),
                        idx,
                    )

            ngc = name_gram_counts[c]

            gram_options = [
                (
                    ngc.get(g, 0),
                    g,
                )
                for g in char3grams(cn)
                if (
                    0 < ngc.get(g, 0)
                    <= gram_max_freq
                )
            ]

            if gram_options:
                gram_options.sort()
                lookup_add(
                    name_gram,
                    (
                        c,
                        gram_options[0][1],
                    ),
                    idx,
                )

            agc = address_gram_counts[c]

            gram_options = [
                (
                    agc.get(g, 0),
                    g,
                )
                for g in char3grams(sa)
                if (
                    0 < agc.get(g, 0)
                    <= gram_max_freq
                )
            ]

            if gram_options:
                gram_options.sort()
                lookup_add(
                    addr_gram,
                    (
                        c,
                        gram_options[0][1],
                    ),
                    idx,
                )

        del df
        gc.collect()

    print("Exact-name keys:", f"{len(exact_name):,}")
    print("Core-name keys:", f"{len(core_name):,}")
    print("Strong-address keys:", f"{len(strong_addr):,}")
    print("Rare-name keys:", f"{len(rare_name):,}")
    print("Rare-address keys:", f"{len(rare_addr):,}")
    print("Name-gram keys:", f"{len(name_gram):,}")
    print("Address-gram keys:", f"{len(addr_gram):,}")

    return (
        exact_name,
        core_name,
        strong_addr,
        rare_name,
        rare_addr,
        name_gram,
        addr_gram,
    )


# static counter used only while building indexes
build_lookups.current_idx = 0


# =========================================================
# PACKED TOP-K
# =========================================================

def pack_entry(score, target_idx):
    q = int(
        max(
            0,
            min(
                1000,
                round(score * 1000),
            ),
        )
    )

    return (
        (q << TARGET_BITS)
        | target_idx
    )


def unpack_score(packed):
    return packed >> TARGET_BITS


def unpack_target_idx(packed):
    return packed & TARGET_MASK


def update_heap(
    heaps,
    s1_idx,
    target_idx,
    matcher_score,
    top_k,
):
    item = pack_entry(
        matcher_score,
        target_idx,
    )

    heap = heaps[s1_idx]

    if heap is None:
        heap = []
        heaps[s1_idx] = heap

    if len(heap) < top_k:
        heapq.heappush(
            heap,
            item,
        )

    elif item > heap[0]:
        heapq.heapreplace(
            heap,
            item,
        )


# =========================================================
# TARGET ID MEMMAP
# =========================================================

def create_target_id_memmap(path, total_rows):
    return np.memmap(
        path,
        dtype=f"S{TARGET_ID_BYTES}",
        mode="w+",
        shape=(total_rows,),
    )


def count_rows(path, chunk_size, limit=None):
    total = 0

    read_kwargs = dict(
        sep="\t",
        usecols=["entity_id"],
        chunksize=chunk_size,
    )
    if limit is not None:
        read_kwargs["nrows"] = limit

    for chunk in pd.read_csv(
        path,
        **read_kwargs,
    ):
        total += len(chunk)
        del chunk

    return total


def get_target_total(paths, chunk_size, limit=None):
    total = 0

    for path in paths:
        n = count_rows(
            path,
            chunk_size,
            limit,
        )

        print(
            f"Rows in {path}: {n:,}"
        )

        total += n

    return total


# =========================================================
# VALIDATION GROUND TRUTH
# =========================================================

def load_validation_truth(
    gt_path,
    validation_ids,
):
    gt = pd.read_csv(
        gt_path,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt = gt[
        gt["source1_entity_id"].isin(
            validation_ids
        )
    ].copy()

    true_by_s1 = {}

    for row in gt.itertuples(index=False):
        if pd.isna(row.matched_entity_ids):
            true_by_s1[row.source1_entity_id] = set()
            continue

        x = str(
            row.matched_entity_ids
        ).strip()

        if not x:
            true_by_s1[row.source1_entity_id] = set()
        else:
            true_by_s1[row.source1_entity_id] = set(
                x.split(",")
            )

    del gt
    gc.collect()

    return true_by_s1


# =========================================================
# PROCESS TARGETS
# =========================================================

def process_targets(
    paths,
    s1_names,
    s1_addresses,
    lookups,
    target_id_memmap,
    heaps,
    args,
    target_limit=None,
):
    (
        exact_name,
        core_name,
        strong_addr,
        rare_name,
        rare_addr,
        name_gram,
        addr_gram,
    ) = lookups

    total_targets = 0
    total_candidates = 0

    for path in paths:
        print("\nScanning:", path)

        processed_file = 0

        read_kwargs = dict(
            sep="\t",
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
            chunksize=args.chunk_size,
        )
        if target_limit is not None:
            read_kwargs["nrows"] = target_limit

        for chunk in pd.read_csv(
            path,
            **read_kwargs,
        ):
            for row in chunk.itertuples(index=False):
                target_idx = total_targets
                total_targets += 1
                processed_file += 1

                target_id = row.entity_id

                target_id_memmap[
                    target_idx
                ] = target_id.encode(
                    "utf-8"
                )

                c = row.country

                n = normalize_text(
                    row.business_name
                )

                a = normalize_text(
                    row.business_address
                )

                cn = strong_name_core(
                    row.business_name
                )

                sa = strong_address(
                    row.business_address
                )

                nt = tokens(n)
                at = tokens(a)

                ng = char3grams(cn)
                ag = char3grams(sa)

                # Candidate union for THIS target only.
                candidate_set = set()

                for idx in lookup_iter(
                    exact_name.get(
                        (c, n)
                    )
                ):
                    candidate_set.add(idx)

                for idx in lookup_iter(
                    core_name.get(
                        (c, cn)
                    )
                ):
                    candidate_set.add(idx)

                for idx in lookup_iter(
                    strong_addr.get(
                        (c, sa)
                    )
                ):
                    candidate_set.add(idx)

                # The S1 lookup dictionaries already contain ONLY
                # rare tokens / eligible rare grams. Therefore we
                # can query them directly; no target-side frequency
                # counters are needed here.
                for t in nt:
                    for idx in lookup_iter(
                        rare_name.get(
                            (c, t)
                        )
                    ):
                        candidate_set.add(idx)

                for t in at:
                    for idx in lookup_iter(
                        rare_addr.get(
                            (c, t)
                        )
                    ):
                        candidate_set.add(idx)

                for g in ng:
                    for idx in lookup_iter(
                        name_gram.get(
                            (c, g)
                        )
                    ):
                        candidate_set.add(idx)

                for g in ag:
                    for idx in lookup_iter(
                        addr_gram.get(
                            (c, g)
                        )
                    ):
                        candidate_set.add(idx)

                total_candidates += len(
                    candidate_set
                )

                # EXACT SAME matcher used in V4.
                for idx in candidate_set:
                    name_score = (
                        fuzz.token_set_ratio(
                            s1_names[idx],
                            n,
                        )
                        / 100.0
                    )

                    address_score = (
                        fuzz.token_set_ratio(
                            s1_addresses[idx],
                            a,
                        )
                        / 100.0
                    )

                    matcher_score = (
                        NAME_WEIGHT * name_score
                        + ADDRESS_WEIGHT * address_score
                    )

                    update_heap(
                        heaps,
                        idx,
                        target_idx,
                        matcher_score,
                        args.top_k,
                    )

            if processed_file % 500_000 < len(chunk):
                print(
                    f"  File progress: "
                    f"{processed_file:,} | "
                    f"total targets: {total_targets:,} | "
                    f"candidate events: {total_candidates:,}",
                    flush=True,
                )

            del chunk
            gc.collect()

        print(
            f"  Done: {processed_file:,}"
        )

    target_id_memmap.flush()

    return (
        total_targets,
        total_candidates,
    )


# =========================================================
# WRITE TEST OUTPUT
# =========================================================

def decode_target_id(memmap, target_idx):
    return bytes(
        memmap[target_idx]
    ).rstrip(
        b"\x00"
    ).decode(
        "utf-8"
    )


def write_test_outputs(
    out_dir,
    s1_ids,
    heaps,
    target_id_memmap,
    threshold,
):
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    matching_path = os.path.join(
        out_dir,
        "matching_results.tsv",
    )

    candidate_path = os.path.join(
        out_dir,
        "candidate_pairs.tsv",
    )

    threshold_q = int(
        round(
            threshold * 1000
        )
    )

    total_candidate_rows = 0
    total_match_rows = 0
    total_matches = 0

    with open(
        candidate_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as fc, open(
        matching_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as fm:
        fc.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        fm.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        for i, sid in enumerate(s1_ids):
            heap = heaps[i]

            if not heap:
                candidate_ids = []
                match_ids = []

            else:
                ranked = sorted(
                    heap,
                    reverse=True,
                )

                candidate_ids = [
                    decode_target_id(
                        target_id_memmap,
                        unpack_target_idx(x),
                    )
                    for x in ranked
                ]

                match_ids = [
                    decode_target_id(
                        target_id_memmap,
                        unpack_target_idx(x),
                    )
                    for x in ranked
                    if unpack_score(x)
                    >= threshold_q
                ]

            fc.write(
                f"{sid}\t"
                f"{','.join(candidate_ids)}\n"
            )

            fm.write(
                f"{sid}\t"
                f"{','.join(match_ids)}\n"
            )

            if candidate_ids:
                total_candidate_rows += 1

            if match_ids:
                total_match_rows += 1
                total_matches += len(
                    match_ids
                )

            if (i + 1) % 100_000 == 0:
                print(
                    f"Wrote "
                    f"{i + 1:,}/"
                    f"{len(s1_ids):,}"
                )

    print("\nOutputs written:")
    print("  ", matching_path)
    print("  ", candidate_path)

    print(
        "\nS1 with candidates:",
        f"{total_candidate_rows:,}",
    )

    print(
        "S1 with matches:",
        f"{total_match_rows:,}",
    )

    print(
        "Predicted matches:",
        f"{total_matches:,}",
    )


# =========================================================
# VALIDATION METRIC
# =========================================================

def entity_f05(true_set, predicted_ids):
    pred = set(predicted_ids)

    tp = len(
        pred & true_set
    )

    p = (
        tp / len(pred)
        if pred
        else (
            1.0
            if not true_set
            else 0.0
        )
    )

    r = (
        tp / len(true_set)
        if true_set
        else (
            1.0
            if not pred
            else 0.0
        )
    )

    if p == 0 and r == 0:
        return 0.0

    return (
        1.25 * p * r
        /
        (
            0.25 * p + r
        )
    )


def score_validation(
    s1_ids,
    heaps,
    target_id_memmap,
    true_by_s1,
    threshold,
):
    threshold_q = int(
        round(
            threshold * 1000
        )
    )

    total = 0.0

    total_true = 0
    total_pred = 0
    total_tp = 0

    singleton_fp = 0

    for i, sid in enumerate(s1_ids):
        heap = heaps[i] or []

        predicted = [
            decode_target_id(
                target_id_memmap,
                unpack_target_idx(x),
            )
            for x in heap
            if unpack_score(x) >= threshold_q
        ]

        true_set = true_by_s1.get(
            sid,
            set(),
        )

        total += entity_f05(
            true_set,
            predicted,
        )

        total_true += len(true_set)
        total_pred += len(predicted)
        total_tp += len(
            set(predicted) & true_set
        )

        if (
            not true_set
            and predicted
        ):
            singleton_fp += 1

    macro_f05 = (
        total / len(s1_ids)
    )

    precision = (
        total_tp / total_pred
        if total_pred
        else 0.0
    )

    recall = (
        total_tp / total_true
        if total_true
        else 0.0
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "MEMORY-SAFE V4 VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        "Validation S1:",
        f"{len(s1_ids):,}",
    )

    print(
        "Total true pairs:",
        f"{total_true:,}",
    )

    print(
        "Predicted matches:",
        f"{total_pred:,}",
    )

    print(
        "True positives:",
        f"{total_tp:,}",
    )

    print(
        "False positives:",
        f"{total_pred - total_tp:,}",
    )

    print(
        "Precision:",
        f"{precision:.4%}",
    )

    print(
        "Pair recall:",
        f"{recall:.4%}",
    )

    print(
        "Singleton false positives:",
        f"{singleton_fp:,}",
    )

    print(
        "MACRO F0.5:",
        f"{macro_f05:.9f}",
    )


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    print("=" * 80)
    print("V4 MEMORY-SAFE")
    print("=" * 80)
    print("Mode:", args.mode)
    print("TOP-K:", args.top_k)
    print("Threshold:", args.threshold)
    print("Gram max frequency:", args.gram_max_freq)
    print("Chunk size:", args.chunk_size)
    if args.limit_s1 is not None:
        print("S1 limit:", args.limit_s1)
    if args.limit_target_rows is not None:
        print("Target-row limit per file:", args.limit_target_rows)
    print()

    if args.mode == "validation":
        s1_path = os.path.join(
            args.train_dir,
            "train_source1.tsv",
        )

        target_paths = [
            os.path.join(
                args.train_dir,
                "train_source2.tsv",
            ),
            os.path.join(
                args.train_dir,
                "train_source3.tsv",
            ),
        ]

        gt_path = os.path.join(
            args.train_dir,
            "train_ground_truth.tsv",
        )

        val_df = pd.read_csv(
            args.validation_ids,
            sep="\t",
            usecols=[
                "source1_entity_id"
            ],
        )

        validation_ids = set(
            val_df[
                "source1_entity_id"
            ]
        )

        del val_df
        gc.collect()

    else:
        s1_path = os.path.join(
            args.test_dir,
            "test_source1.tsv",
        )

        target_paths = [
            os.path.join(
                args.test_dir,
                "test_source2.tsv",
            ),
            os.path.join(
                args.test_dir,
                "test_source3.tsv",
            ),
        ]

        gt_path = None
        validation_ids = None

    # -----------------------------------------------------
    # PASS 1: target frequencies
    # -----------------------------------------------------

    print(
        "PASS 1 — target frequencies"
    )

    (
        name_counts,
        address_counts,
        name_gram_counts,
        address_gram_counts,
    ) = count_target_features(
        target_paths,
        args.chunk_size,
        args.gram_max_freq,
    )

    # -----------------------------------------------------
    # LOAD S1 STRINGS ONLY
    # -----------------------------------------------------

    (
        s1_ids,
        s1_names,
        s1_addresses,
    ) = load_s1(
        s1_path,
        validation_ids,
        args.limit_s1 if args.mode == "test" else None,
    )

    # Reset lookup index counter.
    build_lookups.current_idx = 0

    # -----------------------------------------------------
    # BUILD COMPACT LOOKUPS
    # -----------------------------------------------------

    lookups = build_lookups(
        s1_path,
        validation_ids,
        name_counts,
        address_counts,
        name_gram_counts,
        address_gram_counts,
        args.gram_max_freq,
        args.limit_s1 if args.mode == "test" else None,
    )

    if (
        build_lookups.current_idx
        != len(s1_ids)
    ):
        raise RuntimeError(
            "S1 index mismatch. "
            f"Built {build_lookups.current_idx:,}, "
            f"loaded {len(s1_ids):,}."
        )

    # The blocker lookup keys already encode the rare-token /
    # rare-gram eligibility. We therefore do NOT need the
    # frequency counters during target scanning.
    del name_counts
    del address_counts
    del name_gram_counts
    del address_gram_counts
    gc.collect()

    # -----------------------------------------------------
    # TARGET ID STORAGE
    # -----------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    target_count = get_target_total(
        target_paths,
        args.chunk_size,
        args.limit_target_rows if args.mode == "test" else None,
    )

    target_id_path = os.path.join(
        args.output_dir,
        "target_ids.bin",
    )

    target_id_memmap = create_target_id_memmap(
        target_id_path,
        target_count,
    )

    # -----------------------------------------------------
    # HEAPS
    # -----------------------------------------------------

    print(
        "\nCreating TOP-K storage..."
    )

    heaps = [
        None
    ] * len(s1_ids)

    # Rare-token / rare-gram eligibility was baked into the
    # S1 lookup dictionaries, so no frequency counters are
    # needed during target scanning.

    # -----------------------------------------------------
    # PASS 2
    # -----------------------------------------------------

    print(
        "\nPASS 2 — V4 candidate generation + matcher"
    )

    total_targets, total_candidates = process_targets(
        target_paths,
        s1_names,
        s1_addresses,
        lookups,
        target_id_memmap,
        heaps,
        args,
        args.limit_target_rows if args.mode == "test" else None,
    )

    print(
        "\nTarget rows scanned:",
        f"{total_targets:,}",
    )

    print(
        "Candidate events:",
        f"{total_candidates:,}",
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if args.mode == "validation":
        true_by_s1 = load_validation_truth(
            gt_path,
            validation_ids,
        )

        score_validation(
            s1_ids,
            heaps,
            target_id_memmap,
            true_by_s1,
            args.threshold,
        )

    else:
        # -------------------------------------------------
        # TEST OUTPUT
        # -------------------------------------------------

        write_test_outputs(
            args.output_dir,
            s1_ids,
            heaps,
            target_id_memmap,
            args.threshold,
        )

        print("\nDone.")

    target_id_memmap.flush()


if __name__ == "__main__":
    main()
