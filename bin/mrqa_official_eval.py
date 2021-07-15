"""Official evaluation script for the MRQA Workshop Shared Task.
Adapted fromt the SQuAD v1.1 official evaluation script.
Usage:
    python official_eval.py dataset_file.jsonl.gz prediction_file.json
"""

import argparse
import gzip
import json
import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions


def read_answers(gold_file):
    answers = {}
    if gold_file.endswith(".gz"):
        f = gzip.open(gold_file, "rb")
    else:
        f = open(gold_file)
    for i, line in enumerate(f):
        example = json.loads(line)
        if i == 0 and "header" in example:
            continue
        for qa in example["qas"]:
            answers[qa["qid"]] = qa["answers"]
    f.close()
    return answers


def evaluate(answers, predictions):
    f1 = exact_match = total = 0
    qid_to_scored_predictions = {}
    for qid, ground_truths in answers.items():
        total += 1
        if qid not in predictions:
            message = "Unanswered question %s will receive score 0." % qid
            print(message)
            continue
        prediction = predictions[qid]
        scored_prediction = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        qid_to_scored_predictions[qid] = int(scored_prediction)
        exact_match += scored_prediction

        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {
        "exact_match": exact_match,
        "f1": f1,
        "scores_exact_match": qid_to_scored_predictions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for MRQA Workshop Shared Task")
    parser.add_argument("dataset_file", type=str, help="Dataset File")
    parser.add_argument("prediction_file", type=str, help="Prediction File")
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    answers = read_answers(args.dataset_file)
    predictions = read_predictions(args.prediction_file)
    metrics = evaluate(answers, predictions)

    with open(args.output_file, "w") as f:
        json.dump(metrics, f)
