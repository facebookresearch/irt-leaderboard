"""Copyright (c) Facebook, Inc. and its affiliates."""
# pylint: disable=simplifiable-if-statement
import enum
import glob
import itertools
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import altair as alt
import altair_saver
import pandas as pd
import tqdm
import typer
from dask.distributed import Client
from functional import pseq
from pedroai.io import write_json

from leaderboard.config import conf
from leaderboard.data import IrtParsed, LeaderboardPredictions
from leaderboard.log import get_logger
from leaderboard.rank_stability import get_leaderboard_predictions

log = get_logger(__name__)
comparison_stability_app = typer.Typer()

IRT_PARAMS: List[IrtParsed] = []


class ScoringMethod(enum.Enum):
    RANDOM = "exact_match"
    IRT = "irt"


def simulate(
    *,
    trial_size: int,
    submissions: List[str],
    example_ids: List[str],
    scoring_method: ScoringMethod,
    metric: str = "exact_match",
    fold: str = "dev",
):
    """
    Create simulation based on: http://dx.doi.org/10.1145/564376.564432
    """
    if trial_size * 2 > len(example_ids):
        raise ValueError(f"trial size is too large: {trial_size} * 2 vs {len(example_ids)}")

    if scoring_method == ScoringMethod.IRT and len(IRT_PARAMS) == 0:
        IRT_PARAMS.append(
            IrtParsed.from_irt_file(
                Path(conf["irt"]["squad"][fold]["pyro"]["2PL"]["full"]) / "parameters.json"
            )
        )
    predictions = get_leaderboard_predictions(fold)
    results = []
    submission_combinations = list(itertools.combinations(submissions, 2))
    for sub_x, sub_y in submission_combinations:
        # Setup the trial
        sampled_examples = random.sample(example_ids, 2 * trial_size)
        examples_a = sampled_examples[:trial_size]
        examples_b = sampled_examples[trial_size:]

        sub_x_example_scores = predictions[sub_x][metric]
        sub_y_example_scores = predictions[sub_y][metric]

        # Get scores for submission 1 on subset A and B
        sub_x_a_score = 0
        for example_id in examples_a:
            sub_x_a_score += sub_x_example_scores[example_id]
        sub_x_a_score /= trial_size

        sub_x_b_score = 0
        for example_id in examples_b:
            sub_x_b_score += sub_x_example_scores[example_id]
        sub_x_b_score /= trial_size

        # Get scores for submission 2 on subset A and B
        sub_y_a_score = 0
        for example_id in examples_a:
            sub_y_a_score += sub_y_example_scores[example_id]
        sub_y_a_score /= trial_size

        sub_y_b_score = 0
        for example_id in examples_b:
            sub_y_b_score += sub_y_example_scores[example_id]
        sub_y_b_score /= trial_size

        diff_a = sub_x_a_score - sub_y_a_score
        diff_b = sub_x_b_score - sub_y_b_score

        # If the sign is negative, there was a disagreement
        if diff_a * diff_b < 0:
            swap = True
        else:
            swap = False
        results.append(
            dict(
                trial_size=trial_size,
                x_score=sub_x_a_score,
                y_score=sub_y_a_score,
                diff=diff_a,
                swap=swap,
            )
        )
    return results


def run_stability_analysis(
    step_size: int = 100,
    n_trials: int = 50,
    metric: str = "exact_match",
    scoring_method: ScoringMethod = ScoringMethod.RANDOM,
):
    log.info("Loading data")
    predictions = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"]["dev"])
    submissions = list(predictions.scored_predictions.keys())
    submit_id = submissions[0]
    example_ids = list(predictions.scored_predictions[submit_id][metric].keys())
    max_sample_size = len(example_ids) // 2
    log.info("starting simulation")
    results_by_size = {}
    trial_sizes = list(range(step_size, max_sample_size + step_size, step_size))
    for size in tqdm.tqdm(trial_sizes):
        for _ in range(n_trials):
            if size not in results_by_size:
                results_by_size[size] = []
            results_by_size[size].extend(
                simulate(
                    trial_size=size,
                    submissions=submissions,
                    example_ids=example_ids,
                    metric=metric,
                    scoring_method=scoring_method,
                )
            )

    write_json("data/stability.json", {"results": results_by_size})


CACHED = {}


def compute_parallel(
    job_info: Tuple[int, int], scoring_method: ScoringMethod, metric: str = "exact_match",
):
    size, trial_id = job_info
    output_dir = Path(conf["stability"][scoring_method.value])
    path = output_dir / f"stability_{size}_{trial_id}.json"
    if os.path.exists(path):
        log.info("Simulation output exists, skipping for size=%s", size)
        return size, True
    log.info("Running size=%s", size)
    if "submissions" not in CACHED:
        predictions = LeaderboardPredictions.parse_file(
            conf["squad"]["submission_predictions"]["dev"]
        )
        submissions = list(predictions.scored_predictions.keys())
        submit_id = submissions[0]
        example_ids = list(predictions.scored_predictions[submit_id][metric].keys())
        CACHED["submissions"] = submissions
        CACHED["example_ids"] = example_ids
    else:
        submissions = CACHED["submissions"]
        example_ids = CACHED["example_ids"]
    results = simulate(
        trial_size=size,
        submissions=submissions,
        example_ids=example_ids,
        metric=metric,
        scoring_method=scoring_method,
    )
    df = pd.DataFrame(results)
    df["trial_id"] = trial_id
    df.to_feather(output_dir / f"stability_{size}_{trial_id}.feather")
    return size, True


def run_parallel_simulations(
    *,
    scoring_method: ScoringMethod,
    dask_scheduler: str = "quenya.umiacs.umd.edu:8786",
    step_size: int = 100,
    n_trials: int = 50,
    metric: str = "exact_match",
):
    log.info("creating client")
    client = Client(dask_scheduler)
    log.info("Loading data")
    predictions = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"]["dev"])
    submissions = list(predictions.scored_predictions.keys())
    submit_id = submissions[0]
    example_ids = list(predictions.scored_predictions[submit_id][metric].keys())

    log.info("starting simulation")
    size = step_size
    job_infos = []
    while size < len(example_ids) // 2:
        for t in range(n_trials):
            job_infos.append((size, t))
        size += step_size
    log.info("Parameters: step_size=%s n_example_ids=%s", step_size, len(example_ids))
    job = client.map(
        lambda j: compute_parallel(j, scoring_method=scoring_method, metric=metric), job_infos,
    )
    states = dict(client.gather(job))

    write_json(f"data/stability_{scoring_method.value}.json", {"states": states})


def parse(path: str, bin_size: float):
    trial_df = pd.read_feather(path)
    trial_df["max_score"] = trial_df[["x_score", "y_score"]].max(axis=1)
    trial_df["diff_bin"] = trial_df["diff"].map(lambda x: to_bin(abs(x), bin_size))
    trial_df["max_score_bin"] = trial_df["max_score"].map(lambda x: to_bin(x, bin_size))
    trial_df["n"] = 1
    return trial_df


def to_bin(num, bin_size):
    return math.floor(num / bin_size)


class StabilityPlots:
    def __init__(self, bin_size: float = 0.05) -> None:
        self.bin_size = bin_size
        self.bin_size_percent = int(100 * self.bin_size)
        # TODO: this should reference dev and test
        self.df = pd.concat(
            pseq(glob.glob("data/simulations/stability_*_*.feather"))
            .map(lambda p: parse(p, self.bin_size))
            .list()
        )
        self.grouped_score = (
            self.df.groupby(["trial_size", "max_score_bin"])
            .agg({"swap": "mean", "n": "sum"})
            .reset_index()
        )
        self.grouped_score["max_score_start"] = self.grouped_score["max_score_bin"] * self.bin_size
        self.grouped_score["max_score_start_percent"] = self.grouped_score["max_score_start"].map(
            lambda x: int(x * 100)
        )
        self.grouped_score = self.grouped_score[self.grouped_score["swap"] != 0]

        self.grouped_diff = (
            self.df.groupby(["trial_size", "diff_bin"])
            .agg({"swap": "mean", "n": "sum"})
            .reset_index()
        )
        self.grouped_diff["diff_start"] = self.grouped_diff["diff_bin"] * self.bin_size
        self.grouped_diff["diff_start_percent"] = self.grouped_diff["diff_start"].map(
            lambda x: int(x * 100)
        )
        self.grouped_diff = self.grouped_diff[self.grouped_diff["swap"] != 0]

    def plot_error_by_max_score(self):
        chart = (
            alt.Chart(self.grouped_score)
            .mark_point()
            .encode(
                x=alt.X("trial_size", title="Size of Evaluation Sample"),
                y=alt.Y("swap", title="Comparison Error Rate"),
                color=alt.Color(
                    "max_score_start_percent:N",
                    title=f"Accuracy Bin of Better Model [n, n + {self.bin_size_percent})",
                    legend=alt.Legend(orient="top", titleLimit=400, columns=6),
                ),
                shape=alt.Shape(
                    "max_score_start_percent:N",
                    title=f"Accuracy Bin of Better Model [n, n + {self.bin_size_percent})",
                    legend=alt.Legend(orient="top", titleLimit=400, columns=6),
                ),
                tooltip="max_score_start_percent:N",
            )
            .properties(height=300, width=400)
        )
        return chart

    def plot_error_by_max_score_with_n(self):
        chart = (
            alt.Chart(self.grouped_score)
            .mark_point()
            .encode(
                x=alt.X("trial_size", title="Size of Evaluation Sample"),
                y=alt.Y("swap", title="Comparison Error Rate"),
                shape=alt.Shape(
                    "max_score_start_percent:N",
                    title=["Accuracy Bin of Better Model", f"[n, n + {self.bin_size_percent})",],
                    legend=alt.Legend(columns=2),
                ),
                tooltip="max_score_start_percent:N",
                color=alt.Color(
                    "n",
                    title="Number of Comparisons",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(columns=2),
                ),
            )
        )
        return chart

    def plot_error_by_diff(self):
        chart = (
            alt.Chart(self.grouped_diff)
            .mark_point()
            .encode(
                x=alt.X("trial_size", title="Size of Evaluation Sample"),
                y=alt.Y("swap", title="Comparison Error Rate"),
                color=alt.Color(
                    "diff_start_percent:N",
                    title=f"Accuracy Difference Bin [n, n + {self.bin_size_percent})",
                    legend=alt.Legend(orient="top", titleLimit=400, columns=8),
                ),
                shape=alt.Shape(
                    "diff_start_percent:N",
                    title=f"Accuracy Difference Bin [n, n + {self.bin_size_percent})",
                    legend=alt.Legend(orient="top", titleLimit=400, columns=8),
                ),
                tooltip="diff_start:N",
            )
            .properties(height=300, width=400)
        )
        return chart

    def plot_error_by_diff_with_n(self):
        chart = (
            alt.Chart(self.grouped_diff)
            .mark_point()
            .encode(
                x=alt.X("trial_size", title="Size of Evaluation Sample"),
                y=alt.Y("swap", title="Comparison Error Rate"),
                shape=alt.Shape(
                    "diff_start_percent:N",
                    title=["Accuracy Difference Bin", f"[n, n + {self.bin_size_percent})",],
                    legend=alt.Legend(columns=2),
                ),
                tooltip="diff_start_percent:N",
                color=alt.Color(
                    "n",
                    title="Number of Comparisons",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(columns=2),
                ),
            )
        )
        return chart


def save_chart(chart: alt.Chart, base_path: str):
    for t in [".json", ".svg", ".png", ".html", ".pdf"]:
        path = base_path + t
        altair_saver.save(chart, path)


@comparison_stability_app.command()
def plot():
    log.info("Loading data")
    stability_plots = StabilityPlots()

    log.info("Creating and saving plots")

    save_chart(
        stability_plots.plot_error_by_max_score(), "auto_fig/error_by_max_score",
    )
    save_chart(
        stability_plots.plot_error_by_max_score_with_n(), "auto_fig/error_by_max_score_with_n",
    )
    save_chart(stability_plots.plot_error_by_diff(), "auto_fig/error_by_diff")
    save_chart(stability_plots.plot_error_by_diff_with_n(), "auto_fig/error_by_diff_with_n")
