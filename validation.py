import pandas as pd


def parse_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()

    return set(str(value).split(","))


def f05_score(true_ids, pred_ids):

    # Correct singleton prediction
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    # No predictions
    if len(pred_ids) == 0:
        return 0.0

    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(ground_truth, predictions):

    # Fast dictionary lookup
    pred_map = dict(
        zip(
            predictions["source1_entity_id"],
            predictions["matched_entity_ids"]
        )
    )

    scores = []

    for row in ground_truth.itertuples(index=False):

        s1_id = row.source1_entity_id

        true_ids = parse_ids(row.matched_entity_ids)
        pred_ids = parse_ids(pred_map.get(s1_id, ""))

        scores.append(f05_score(true_ids, pred_ids))

    return sum(scores) / len(scores)