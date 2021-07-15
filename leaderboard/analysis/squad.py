"""Copyright (c) Facebook, Inc. and its affiliates."""
import copy
import datetime
import os
from pathlib import Path
from typing import List

import pandas as pd
import plotnine as p9
from mlxtend.evaluate import permutation_test
from pedroai.io import read_json
from pedroai.plot import theme_pedroai
from plotnine import aes, facet_wrap, geom_bar, ggplot
from statsmodels.stats.anova import AnovaRM

from leaderboard.config import conf
from leaderboard.log import get_logger
from leaderboard.squad_eval_v2 import get_raw_scores

log = get_logger(__name__)


def load_squad():
    dataset = read_json(conf["squad"]["dev_v2"])["data"]
    question_map = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for question in p["qas"]:
                question_map[question["id"]] = question["is_impossible"]
    return dataset, question_map


def load_machine_predictions(dataset, models: List[str]):
    exact_scores = {}
    f1_scores = {}
    for name in models:
        if name == "human":
            continue
        preds = read_json(f"data/squad/{name}.json")
        exact_scores[name], f1_scores[name] = get_raw_scores(dataset, preds)
    return exact_scores, f1_scores


def load_human_predictions(dataset):
    human_preds = read_json("data/squad/human-pred-dev.json")
    exact_scores, f1_scores = get_raw_scores(dataset, human_preds)
    return exact_scores, f1_scores


FIG_PATH = Path("auto_fig")
PNG_OUT = bool(os.environ.get("PNG_OUT", False))
PDF_OUT = bool(os.environ.get("PDF_OUT", True))
COLORS = ["#45a9b5", "#D11149", "#87ff9f", "#F17105", "#E6C229"]


def save(plot, filename, custom_theme=None, **kwargs):
    if custom_theme is None:
        plot = plot + theme_pedroai()
    else:
        plot = plot + theme_pedroai() + custom_theme

    if PNG_OUT:
        plot.save(FIG_PATH / f"{filename}.png", **kwargs)
    if PDF_OUT:
        plot.save(FIG_PATH / f"{filename}.pdf", **kwargs)


class SquadPredictionData:
    def __init__(self, models: List[str]):
        self.models = models
        self.dataset, self.question_map = load_squad()
        self.machine_exact, self.machine_f1 = load_machine_predictions(self.dataset, self.models)
        self.human_exact, self.human_f1 = load_human_predictions(self.dataset)
        self.exact_scores = copy.deepcopy(self.machine_exact)
        self.exact_scores["human"] = self.human_exact
        self.f1_scores = copy.deepcopy(self.machine_f1)
        self.f1_scores["human"] = self.human_f1

    def plot_comparison(self):
        rows = []
        for qid in self.question_map:
            n_correct = 0
            n_total = 0
            correct_models = []
            for name in self.models:
                correct = self.exact_scores[name][qid]
                if correct == 1:
                    correct_models.append(name)
                n_correct += correct
                n_total += 1
            if self.exact_scores["human"][qid] == 1:
                human_result = "Human-Correct"
            else:
                human_result = "Human-Wrong"
            if len(correct_models) == 0:
                correct_models = ["none"]
            rows.append(
                {
                    "qid": qid,
                    "n_correct": n_correct,
                    "p_correct": n_correct / n_total,
                    "human": human_result,
                    "name": " ".join(correct_models) + " : " + human_result,
                    "is_impossible": self.question_map[qid],
                }
            )
        df = pd.DataFrame(rows)
        df["n_correct"] = pd.Categorical(df["n_correct"])
        p = ggplot(df) + aes(x="n_correct", fill="name") + facet_wrap("is_impossible") + geom_bar()
        save(p, "squad_diff")

    def repeated_measures_anova(self):
        rows = []
        for qid in self.question_map:
            for model in self.models:
                correct = self.exact_scores[model][qid]
                if correct == 1:
                    correct = 1
                else:
                    correct = 0

                rows.append({"qid": qid, "model": model, "correct": correct})
        df = pd.DataFrame(rows)
        log.info("Models: %s", self.models)
        log.info("DF Stats: %s", df.describe())
        log.info(
            "%s", AnovaRM(data=df, depvar="correct", subject="qid", within=["model"]).fit(),
        )

    def permutation_test(self):
        if len(self.models) != 2:
            raise ValueError("Permutation test expects two models")
        rows = []
        for qid in self.question_map:
            for model in self.models:
                correct = self.exact_scores[model][qid]
                if correct == 1:
                    correct = 1
                else:
                    correct = 0

                rows.append({"qid": qid, "model": model, "correct": correct})
        # model is the treatment, correct is the outcome
        df = pd.DataFrame(rows)
        control = df[df.model == self.models[0]].correct
        treatment = df[df.model == self.models[1]].correct
        p_value = permutation_test(
            treatment, control, method="approximate", num_rounds=10000, seed=42
        )
        log.info("P Value: %s", p_value)

    def dist_info(self):
        rows = []
        for qid in self.question_map:
            for model in self.models:
                correct = self.exact_scores[model][qid]
                if correct == 1:
                    correct = 1
                    label = "correct"
                else:
                    correct = 0
                    label = "wrong"

                rows.append({"qid": qid, "model": model, "correct": correct, "label": label})
        # model is the treatment, correct is the outcome
        df = pd.DataFrame(rows)
        log.info(df.head())
        log.info(df.pivot(index="qid", columns="model", values="correct").head())
        log.info(df.pivot(index="qid", columns="model", values="correct").corr())
        log.info("%s", df.groupby("model").describe())
        log.info("%s", df.groupby("model").agg({"correct": [pd.Series.kurt, pd.Series.skew]}))

    def to_df(self):
        rows = []
        for qid in self.question_map:
            for model in self.models:
                correct = self.exact_scores[model][qid]
                if correct == 1:
                    correct = 1
                    label = "correct"
                else:
                    correct = 0
                    label = "wrong"

                rows.append({"qid": qid, "model": model, "correct": correct, "label": label})
        # model is the treatment, correct is the outcome
        return pd.DataFrame(rows)

    def to_csv(self, path: str):
        # model is the treatment, correct is the outcome
        df = self.to_df()
        df.pivot(index="qid", columns="model", values="correct").to_csv(path)


class LeaderboardData:
    def __init__(self):
        self._data = read_json("data/out-v2.0.json")
        self._submissions = []
        for s in self._data["leaderboard"]:
            name = s["submission"]["description"]
            created = s["submission"]["created"]
            self._submissions.append(
                {
                    "name": name,
                    "created": datetime.datetime.fromtimestamp(created),
                    "metric": "exact_match",
                    "value": s["scores"]["exact_match"],
                }
            )
            self._submissions.append(
                {
                    "name": name,
                    "created": datetime.datetime.fromtimestamp(created),
                    "metric": "f1",
                    "value": s["scores"]["f1"],
                }
            )
        self._df = pd.DataFrame(self._submissions)

    def plot(self):
        df = self._df.dropna()
        df["is_ensemble"] = df["name"].map(lambda x: "ensemble" in x)
        return (
            ggplot(df)
            + aes(x="created", y="value", color="is_ensemble")
            + facet_wrap("metric", nrow=2)
            + p9.geom_point()
            + p9.theme(figure_size=(12, 9), axis_text_x=p9.element_text(angle=45))
            + p9.labs(x="Submission Time", y="Score")
        )
