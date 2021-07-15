"""
Copyright (c) Facebook, Inc. and its affiliates.
Create lists of manual or model-based features to derive correlations to
difficulty and discriminability with.
"""
import enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import altair as alt
import numpy as np
import typer
from pedroai.io import read_json, safe_file
from rich.console import Console
from rich.progress import track

from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import (
    IrtParsed,
    LeaderboardPredictions,
    LeaderboardSplits,
    load_squad_v2,
)
from leaderboard.irt.evaluate import evaluate_item_predictions
from leaderboard.topics import TopicModel

alt.data_transformers.disable_max_rows()
console = Console()


features_app = typer.Typer()


@features_app.command()
def to_vw(*, irt_family: str, irt_type: str, feature_set: str, out_dir: str):
    out_dir = Path(out_dir)
    if irt_family != "no_irt":
        irt_param_file = (
            DATA_ROOT / conf["irt"]["squad"][irt_family][irt_type]["full"] / "parameters.json"
        )
    else:
        irt_param_file = None
    features: List[str] = conf["vw"][feature_set]["features"]
    parameters = conf["vw"][feature_set]["parameters"]
    vw_dataset = VWDataset(
        train_file=safe_file(out_dir / "train.vw.txt"),
        test_file=safe_file(out_dir / "test.vw.txt"),
        irt_param_file=irt_param_file,
        features=features,
        # We only train the VW linear model on squad dev, since we don't have the text
        # of the test set
        train_test_split_file=DATA_ROOT / conf["squad"]["leaderboard_splits"]["dev"],
        parameters=parameters,
    )
    vw_dataset.write_examples()


@features_app.command()
def eval_vw(
    test_examples_file: str,
    test_pred_file: str,
    report_dir: str,
    threshold: float = 0.5,
    name: str = "vw",
):
    report_dir = Path(report_dir)
    with open(test_examples_file) as ex_f, open(test_pred_file) as pred_f:
        n = 0
        correct = 0
        label_dist = {True: 0, False: 0}
        test_labels = []
        test_probs = []
        test_preds = []
        for ex_line, pred_line in zip(ex_f, pred_f):
            label = int(ex_line.split()[0])
            if label == 1:
                label = True
            elif label == -1:
                label = False
            else:
                raise ValueError(f"Invalid label: {label}")
            test_labels.append(label)

            p = float(pred_line)
            if p < 0 or p > 1:
                raise ValueError(f"Invalid prob: {p}")
            test_probs.append(p)
            pred_label = p >= threshold
            test_preds.append(pred_label)
            label_dist[label] += 1
            if label == pred_label:
                correct += 1
            n += 1
        score = correct / n
        majority = max(v for v in label_dist.values()) / n
        console.log(
            f"N={n} Correct={correct} Model Acc: {score} Baseline_Acc: {majority} Dist={label_dist}"
        )
        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)
        evaluate_item_predictions(
            report_dir=report_dir, pred_probs=test_probs, labels=test_labels, name=name
        )


def load_sugawara_squad2_features():
    easy_questions = list(
        read_json(DATA_ROOT / "data/squad/datasets/squad-easy-subset.json").keys()
    )
    hard_questions = list(
        read_json(DATA_ROOT / "data/squad/datasets/squad-hard-subset.json").keys()
    )

    hardness = {}
    for q in easy_questions:
        hardness[q] = "easy"
    for q in hard_questions:
        hardness[q] = "hard"
    return hardness


def vw_escape(text: str) -> str:
    return text.replace("|", " ").replace(":", " ").replace("\n", " ")


class Feature(enum.Enum):
    IRT = "irt"
    GUIDS = "guids"
    QWORDS = "qwords"
    CWORDS = "cwords"
    STATS = "stats"
    TITLE = "title"
    BASELINE = "baseline"
    M_ID = "m_id"
    EX_ID = "ex_id"
    TOPICS = "topics"


def create_vw_irt_features(
    *, diff: float, disc: Optional[float], ability: float, lambda_: Optional[float]
):
    irt_features = []
    irt_features.append(f"diff:{diff}")
    irt_features.append(f"ability:{ability}")
    if disc is not None:
        irt_features.append(f"disc:{disc}")

    if lambda_ is not None:
        irt_features.append(f"lambda:{lambda_}")
    feature_str = " ".join(irt_features)
    return f"|irt {feature_str}"


class VWDataset:
    def __init__(
        self,
        *,
        train_file: Union[str, Path],
        test_file: Union[str, Path],
        irt_param_file: Optional[Union[str, Path]],
        features: List[str],
        train_test_split_file: str,
        parameters: Optional[Dict],
    ) -> None:
        super().__init__()
        self._train_file = train_file
        self._test_file = test_file
        self._features = features
        self._feature_set = set(Feature(f) for f in features)
        if irt_param_file is None:
            self._irt_param_file = None
            self._irt = None
        else:
            self._irt_param_file = irt_param_file
            self._irt = IrtParsed.from_irt_file(irt_param_file)
        # We only have disc/diff parameters for dev questions
        self._predictions = LeaderboardPredictions.parse_file(
            DATA_ROOT / conf["squad"]["submission_predictions"]["dev"]
        )

        console.log("Loading Squad Dev Dataset")
        self._dev_squad = load_squad_v2(conf["squad"]["dev_v2"])
        if Feature.TOPICS in self._feature_set:
            console.log("Loading topic model")
            self._topic_model = TopicModel.load(
                conf["topic"][parameters["topic_name"]]["output_dir"]
            )
        else:
            self._topic_model = None
        self._hardness = load_sugawara_squad2_features()
        self._splits = LeaderboardSplits.parse_file(train_test_split_file)
        self._train_items = {(r.model_id, r.example_id) for r in self._splits.train}
        self._test_items = {(r.model_id, r.example_id) for r in self._splits.test}

    def write_examples(self):
        with open(self._train_file, "w") as train_f, open(self._test_file, "w") as test_f:
            for fold, example in self.generate_examples():
                if fold == "train":
                    train_f.write(f"{example}\n")
                elif fold == "test":
                    test_f.write(f"{example}\n")
                else:
                    raise ValueError("invalid fold")

    def generate_examples(self):
        label_dist = {1: 0, -1: 0}
        for page in track(self._dev_squad.data):
            for paragraph in page.paragraphs:
                for question in paragraph.qas:
                    if self._irt is None:
                        example_stats = None
                    else:
                        example_stats = self._irt.example_stats[question.id]
                    if self._topic_model is None:
                        vw_topic_feature = None
                    else:
                        vw_topic_feature = self._topic_model.vw_features(question.id)
                    if len(question.answers) == 0:
                        answer_position = 0
                        answer_length = 0
                    else:
                        answer_position = min(a.answer_start for a in question.answers)
                        answer_length = min(len(a.text) for a in question.answers)

                    if question.id in self._hardness:
                        hardness = self._hardness[question.id]
                    else:
                        # This happens for impossible questions
                        hardness = "NA"
                    for model_id, preds in self._predictions.scored_predictions.items():
                        if self._irt is None:
                            model_stats = None
                        else:
                            model_stats = self._irt.model_stats[model_id]
                        em_scores = preds["exact_match"]
                        score = int(em_scores[question.id])
                        c_char_length = len(paragraph.context)
                        c_word_length = len(paragraph.context.split())
                        q_char_length = len(question.question)
                        q_word_length = len(question.question.split())
                        if score == 1:
                            label = 1
                        elif score == 0:
                            label = -1
                        else:
                            raise ValueError("Invalid label")
                        label_dist[label] += 1
                        text = vw_escape(question.question)
                        features = []
                        features.append(f"{label}")
                        if Feature.GUIDS in self._feature_set:
                            features.append(f"|ids {question.id} {model_id}")
                        if Feature.M_ID in self._feature_set:
                            features.append(f"|m_id {model_id}")
                        if Feature.EX_ID in self._feature_set:
                            features.append(f"|ex_id {question.id}")
                        if Feature.QWORDS in self._feature_set:
                            features.append(f"|question {text}")
                        if Feature.STATS in self._feature_set:
                            q_word_length = len(question.question.split())
                            if question.is_impossible:
                                vw_impossible = "impossible"
                            else:
                                vw_impossible = "possible"

                            stats_features = [
                                "|stats",
                                f"q_word_length:{q_word_length}",
                                f"q_char_length:{q_char_length}",
                                f"answer_position:{answer_position}",
                                f"answer_char_length:{answer_length}",
                                f"c_char_length:{c_char_length}",
                                f"c_word_length:{c_word_length}",
                                vw_impossible,
                                hardness,
                            ]
                            features.append(" ".join(stats_features))
                        if Feature.CWORDS in self._feature_set:
                            context_text = vw_escape(paragraph.context)
                            features.append(f"|context {context_text}")
                        if Feature.TITLE in self._feature_set:
                            article_text = vw_escape(page.title)
                            features.append(f"|title {article_text}")
                        if Feature.IRT in self._feature_set:
                            features.append(
                                create_vw_irt_features(
                                    diff=example_stats.diff,
                                    disc=example_stats.disc,
                                    ability=model_stats.skill,
                                    lambda_=example_stats.lambda_,
                                )
                            )
                        if Feature.BASELINE in self._feature_set:
                            features.append("|baseline a")
                        if Feature.TOPICS in self._feature_set:
                            if vw_topic_feature is None:
                                raise ValueError("Invalid topic feature")
                            else:
                                features.append(vw_topic_feature)
                        example = " ".join(features)
                        if (model_id, question.id) in self._train_items:
                            yield "train", example
                        elif (model_id, question.id) in self._test_items:
                            yield "test", example
                        else:
                            raise ValueError(
                                f"Item not in train or test: {(model_id, question.id)}"
                            )
        console.log("Label Distribution")
        total = sum(label_dist.values())
        console.log(label_dist)
        console.log({k: v / total for k, v in label_dist.items()})
        console.log("dataset summary")
