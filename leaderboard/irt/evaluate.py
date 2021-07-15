"""
Copyright (c) Facebook, Inc. and its affiliates.
We evaluate IRT models in two ways:
- Given all the item information, how accurate is the trained model on the same items?
- Given 90% of the items for training, how accurate is the trained model on the remaining 10%
"""
from pathlib import Path
from typing import Dict, List

import altair as alt
import altair_saver
import numpy as np
import pandas as pd
from pedroai.io import write_json
from rich.console import Console
from rich.progress import track
from sklearn import metrics

from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import (
    IrtParsed,
    LeaderboardPredictions,
    LeaderboardSplits,
    MultidimIrtParsed,
)

alt.data_transformers.disable_max_rows()
console = Console()

Vector = List[float]


def irt_multi_2pl(*, ability: Vector, diff: Vector, disc: Vector):
    ability = np.array(ability)
    diff = np.array(diff)
    disc = np.array(disc)
    logit = -disc * (ability - diff)
    return 1 / (1 + np.exp(logit.sum()))


def irt_1pl(*, ability: float, diff: float):
    return 1 / (1 + np.exp(-(ability - diff)))


def irt_2pl(*, ability: float, diff: float, disc: float):
    return 1 / (1 + np.exp(-disc * (ability - diff)))


def irt_3pl(*, ability: float, diff: float, disc: float, lambda_: float):
    return lambda_ / (1 + np.exp(-disc * (ability - diff)))


def evaluate_irt_model(*, evaluation: str, model_family: str, model_type: str, fold: str):
    irt_base_dir = DATA_ROOT / conf["irt"]["squad"][fold][model_family][model_type][evaluation]
    evaluate_irt_model_in_dir(
        irt_base_dir=irt_base_dir,
        fold=fold,
        evaluation=evaluation,
        model_type=model_type,
        model_family=model_family,
    )


def evaluate_multidim_model_in_dir(
    *, irt_base_dir: str, fold: str, evaluation: str, model_type: str, model_family: str
):
    # pylint: disable=unreachable
    parameters = MultidimIrtParsed.from_irt_file(irt_base_dir / "parameters.json")
    predictions = LeaderboardPredictions.parse_file(
        DATA_ROOT / conf["squad"]["submission_predictions"][fold]
    )
    if evaluation == "heldout":
        splits = LeaderboardSplits.parse_file(conf["squad"]["leaderboard_splits"][fold])
        test_items = {(r.model_id, r.example_id) for r in splits.test}

        def use_item(model_id: str, example_id: str):
            return (model_id, example_id) in test_items

    elif evaluation == "full":

        def use_item(model_id: str, example_id: str):
            # pylint: disable=unused-argument
            return True

    else:
        raise ValueError(f"Invalid irt evaluation type: {evaluation}")

    pred_probs = []
    pred_labels = []
    labels = []
    for model_id, preds in track(predictions.scored_predictions.items()):
        em_scores = preds["exact_match"]
        for example_id, score in em_scores.items():
            if not use_item(model_id, example_id):
                continue
            score = int(score)
            example_stats = parameters.example_stats[example_id]
            model_stats = parameters.model_stats[model_id]
            if model_type == "MD1PL":
                prob = irt_multi_2pl(
                    ability=model_stats.skill, diff=example_stats.diff, disc=example_stats.disc,
                )
            elif model_type == "2PL":
                raise NotImplementedError()
                prob = irt_2pl(
                    ability=model_stats.skill, diff=example_stats.diff, disc=example_stats.disc,
                )
            elif model_type == "3PL":
                raise NotImplementedError()
                prob = irt_3pl(
                    ability=model_stats.skill,
                    diff=example_stats.diff,
                    disc=example_stats.disc,
                    lambda_=example_stats.lambda_,
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            pred = 1 if prob > 0.5 else 0
            pred_probs.append(prob)
            pred_labels.append(pred)
            labels.append(score)

    pred_probs = np.array(pred_probs)
    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    name = f"{model_family}-{model_type}-{evaluation}"
    evaluate_item_predictions(
        report_dir=irt_base_dir, pred_probs=pred_probs, labels=labels, name=name
    )


def evaluate_irt_model_in_dir(
    *, irt_base_dir: str, fold: str, evaluation: str, model_type: str, model_family: str
):
    """Compute accuracy of IRT Model

    Args:
        model_family (str): pyro or stan
        model_type (str): 1PL, 2PL, 3PL, etc
        evalaution (str): Whether to compute on all items or only held out items

    Raises:
        ValueError: On invalid model family/type
    """
    parameters = IrtParsed.from_irt_file(irt_base_dir / "parameters.json")
    predictions = LeaderboardPredictions.parse_file(
        DATA_ROOT / conf["squad"]["submission_predictions"][fold]
    )
    if evaluation == "heldout":
        splits = LeaderboardSplits.parse_file(conf["squad"]["leaderboard_splits"][fold])
        test_items = {(r.model_id, r.example_id) for r in splits.test}

        def use_item(model_id: str, example_id: str):
            return (model_id, example_id) in test_items

    elif evaluation == "full":

        def use_item(model_id: str, example_id: str):
            # pylint: disable=unused-argument
            return True

    else:
        raise ValueError(f"Invalid irt evaluation type: {evaluation}")

    pred_probs = []
    pred_labels = []
    labels = []
    for model_id, preds in track(predictions.scored_predictions.items()):
        em_scores = preds["exact_match"]
        for example_id, score in em_scores.items():
            if not use_item(model_id, example_id):
                continue
            score = int(score)
            example_stats = parameters.example_stats[example_id]
            model_stats = parameters.model_stats[model_id]
            if model_type == "1PL":
                prob = irt_1pl(ability=model_stats.skill, diff=example_stats.diff)
            elif model_type == "2PL":
                prob = irt_2pl(
                    ability=model_stats.skill, diff=example_stats.diff, disc=example_stats.disc,
                )
            elif model_type == "3PL":
                prob = irt_3pl(
                    ability=model_stats.skill,
                    diff=example_stats.diff,
                    disc=example_stats.disc,
                    lambda_=example_stats.lambda_,
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            pred = 1 if prob > 0.5 else 0
            pred_probs.append(prob)
            pred_labels.append(pred)
            labels.append(score)

    pred_probs = np.array(pred_probs)
    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    name = f"{model_family}-{model_type}-{evaluation}"
    evaluate_item_predictions(
        report_dir=irt_base_dir, pred_probs=pred_probs, labels=labels, name=name
    )


def evaluate_item_predictions(
    *,
    name: str,
    report_dir: Path,
    pred_probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    results = {"name": name}
    pred_labels = pred_probs > threshold
    correct = int((labels == pred_labels).sum())
    accuracy = correct / len(labels)
    results["accuracy"] = {
        "n": len(labels),
        "correct": correct,
        "score": accuracy,
    }
    roc_auc = metrics.roc_auc_score(labels, pred_probs)
    console.log(f"ROC AUC: {roc_auc}")
    results["roc_auc"] = roc_auc

    console.log("Report")
    classification_report = metrics.classification_report(labels, pred_labels, output_dict=True)
    console.log(classification_report)
    results["classification_report"] = classification_report

    console.log("Confusion Matrix")
    confusion_matrix = metrics.confusion_matrix(labels, pred_labels)
    console.log(confusion_matrix)
    results["confusion_matrix"] = confusion_matrix.tolist()
    write_json(report_dir / "report.json", results)

    fpr, tpr, thresholds = metrics.roc_curve(labels, pred_probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    if len(roc_df) > 5000:
        roc_df = roc_df.sample(5000)
    roc_chart = (
        alt.Chart(roc_df)
        .mark_point()
        .encode(x="fpr", y="tpr", color="thresholds", size=alt.value(1))
    )
    altair_saver.save(roc_chart, str(report_dir / "roc.pdf"), method="node")

    precision, recall, thresholds = metrics.precision_recall_curve(labels, pred_probs)
    # The final entry of precision=1, recall=0, but has no true threshold
    thresholds = np.append(thresholds, [1])
    pr_df = pd.DataFrame({"precision": precision, "recall": recall, "thresholds": thresholds})
    if len(pr_df) > 5000:
        pr_df = pr_df.sample(5000)
    pr_chart = alt.Chart(pr_df).mark_point().encode(x="recall", y="precision", size=alt.value(1))
    altair_saver.save(pr_chart, str(report_dir / "precision_recall.pdf"), method="node")
