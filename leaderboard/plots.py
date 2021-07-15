"""Copyright (c) Facebook, Inc. and its affiliates."""
# pylint: disable=unused-argument,too-many-statements,unused-variable
import functools
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import altair as alt
import altair_saver
import numpy as np
import pandas as pd
import typer
from altair.expr import datum
from functional import pseq, seq
from pedroai.io import (
    read_json,
    read_jsonlines,
    requires_file,
    requires_files,
    safe_file,
)
from pedroai.math import to_precision
from rich.console import Console

from leaderboard.config import conf
from leaderboard.data import (
    IrtParsed,
    LeaderboardPredictions,
    load_squad_submissions,
    load_squad_v2,
)

alt.data_transformers.disable_max_rows()

PAPERS_ROOT = Path(os.environ.get("PAPERS_ROOT", "./"))
AUTO_FIG = PAPERS_ROOT / "auto_fig"
COMMIT_AUTO_FIGS = PAPERS_ROOT / "commit_auto_figs"
BASE_SIZE = 150
plot_app = typer.Typer()
console = Console()


def save_chart(chart: alt.Chart, base_path: Union[str, Path], filetypes: List[str]):
    if isinstance(base_path, Path):
        base_path = str(base_path)
    for t in filetypes:
        path = base_path + "." + t
        if t in ("svg", "pdf"):
            method = "node"
        else:
            method = None
        console.log(f"Saving to: {path}")
        altair_saver.save(chart, safe_file(path), method=method)


def generate_ablation_files():
    ablation_files = {}
    for path in glob.glob("data/linear/**/**/**/report.json"):
        fields = path.split("/")
        irt_family = fields[2]
        irt_type = fields[3]
        features = fields[4]
        if irt_type in ("1PL", "2PL"):
            continue
        ablation_files[(irt_family, irt_type, features)] = Path(path)
    return ablation_files


PLOTS = {}


def register_plot(name: str):
    def decorator(func):
        PLOTS[name] = func
        return func

    return decorator


ABLATION_FILES = generate_ablation_files()


def generate_irt_files():
    irt_files = {}
    for model_type, evaluations in conf["irt"]["squad"]["dev"]["pyro"].items():
        for eval_type in ("full", "heldout"):
            irt_files[(model_type, eval_type)] = Path(evaluations[eval_type]) / "report.json"
    return irt_files


IRT_FILES = generate_irt_files()


def init_score():
    return {"tie": 0, "win": 0, "loss": 0}


def run_stats_tournament(fold: str):
    test_results = {}
    for test in ["mcnemar", "see", "sem", "student_t", "wilcoxon"]:
        stats = read_json(f"data/stats/fold={fold}/sampling=random/percent=100/{test}.json")
        match_results = defaultdict(init_score)
        alpha = 0.01
        for r in stats["results"]:
            model_a = r["model_a"]
            model_b = r["model_b"]
            if r["pvalue"] is not None and r["pvalue"] < alpha:
                if r["score_a"] > r["score_b"]:
                    match_results[model_a]["win"] += 1
                    match_results[model_b]["loss"] += 1
                else:
                    match_results[model_a]["loss"] += 1
                    match_results[model_b]["win"] += 1
            else:
                match_results[model_a]["tie"] += 1
                match_results[model_b]["tie"] += 1
        test_results[test] = match_results
    return test_results


@register_plot("rank_correlation_table")
@requires_file(conf["squad"]["dev_to_test"])
def rank_correlation_table(filetypes: List[str], commit: bool = False, include_test: bool = True):
    irt_model = "3PL"
    dev_irt_params = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["dev"]["pyro"][irt_model]["full"]) / "parameters.json"
    )
    dev_predictions = LeaderboardPredictions.parse_file(
        conf["squad"]["submission_predictions"]["dev"]
    )
    dev_id_to_subject = load_squad_submissions(dev_predictions)
    console.log("N Dev IRT", len(dev_irt_params.model_stats))

    stats_results = run_stats_tournament("dev")
    mcnemar_results = stats_results["mcnemar"]
    see_results = stats_results["see"]
    student_t_results = stats_results["student_t"]
    sem_results = stats_results["sem"]

    if include_test:
        mapping = read_json(conf["squad"]["dev_to_test"])
        dev_to_test = mapping["dev_to_test"]
        test_irt_params = IrtParsed.from_irt_file(
            Path(conf["irt"]["squad"]["test"]["pyro"][irt_model]["full"]) / "parameters.json"
        )
        console.log("N Test IRT", len(test_irt_params.model_stats))
        test_stats_results = run_stats_tournament("test")
        test_mcnemar_results = test_stats_results["mcnemar"]
        test_see_results = test_stats_results["see"]
        test_student_t_results = test_stats_results["student_t"]
        test_sem_results = test_stats_results["sem"]
    else:
        mapping = None
        dev_to_test = None
        test_irt_params = None
        test_stats_results = None
        test_mcnemar_results = None
        test_see_results = None
        test_student_t_results = None
        test_sem_results = None

    rows = []
    n_test = 0
    n_dev = 0
    for subject_id in dev_id_to_subject.keys():
        subject = dev_id_to_subject[subject_id]
        entry = {
            "subject_id": subject_id,
            "name": subject["name"],
            "dev_em": subject["dev_em"],
            "test_em": subject["test_em"],
            "dev_skill": dev_irt_params.model_stats[subject_id].skill,
            # "dev_mcnemar": mcnemar_results[subject_id]["win"],
            # "dev_see": see_results[subject_id]["win"],
            # "dev_student_t": student_t_results[subject_id]["win"],
            # "dev_sem": sem_results[subject_id]["win"],
        }
        n_dev += 1
        if include_test:
            if subject_id in dev_to_test:
                test_subject_id = dev_to_test[subject_id]
                if test_subject_id in test_irt_params.model_stats:
                    entry["test_skill"] = test_irt_params.model_stats[test_subject_id].skill
                    # entry["test_mcnemar"] = test_mcnemar_results[test_subject_id]["win"]
                    # entry["test_see"] = test_see_results[test_subject_id]["win"]
                    # entry["test_student_t"] = test_student_t_results[test_subject_id][
                    #     "win"
                    # ]
                    # entry["test_sem"] = test_sem_results[test_subject_id]["win"]
                    n_test += 1
        rows.append(entry)

    console.log("N Dev", n_dev, "N Test", n_test)
    df = pd.DataFrame(rows).dropna(axis=0)
    console.log(df)

    name_mapping = {
        "dev_em": r"EM$_{\text{dev}}$",
        "test_em": r"EM$_{\text{test}}$",
        "dev_skill": r"Ability$_{\text{dev}}$",
        "test_skill": r"Ability$_{\text{test}}$",
    }
    correlations = df.corr(method="kendall")
    correlations.to_pickle("/tmp/leaderboard_correlations.pickle")
    console.log(correlations)
    print(
        correlations.applymap(lambda n: f"${to_precision(n, 3)}$")
        .rename(columns=name_mapping, index=name_mapping)
        .to_latex(column_format="l" + len(name_mapping) * "r", escape=False)
    )


@register_plot("sampling_stability")
def sample_stability_plot(filetypes: List[str], commit: bool = False):
    input_dir = Path(conf["stability"]["sampling"])
    random_df = pd.read_json(input_dir / "random_df.json")
    irt_df = pd.read_json(input_dir / "irt_df.json")
    info_df = pd.read_json(input_dir / "info_df.json")

    method_names = {
        "dev_high_disc_to_test": "High Discrimination",
        "dev_high_diff_to_test": "High Difficulty",
        "dev_high_disc_diff_to_test": "High Disc + Diff",
        "dev_info_to_test": "High Information",
        "dev_random_to_test": "Random",
    }

    def format_df(dataframe):
        return dataframe.assign(
            sampling_method=dataframe["variable"].map(lambda v: method_names[v])
        )

    x_scale = alt.X("trial_size", title="Development Set Sample Size", scale=alt.Scale(type="log"))
    y_scale = alt.Scale(zero=False)
    color_scale = alt.Color(
        "sampling_method",
        title="Sampling Method",
        legend=alt.Legend(orient="bottom-right", fillColor="white", padding=5, strokeColor="gray"),
        sort=[
            "High Disc + Diff",
            "High Information",
            "High Discrimination",
            "High Difficulty",
            "Random",
        ],
    )
    random_line = (
        alt.Chart(format_df(random_df))
        .mark_line()
        .encode(
            x=x_scale,
            y=alt.Y("mean(value)", scale=y_scale, title="Correlation to Test Rank"),
            color=color_scale,
        )
    )
    random_band = (
        alt.Chart(format_df(random_df))
        .mark_errorband(extent="ci")
        .encode(x=x_scale, y=alt.Y("value", title="", scale=y_scale), color=color_scale)
    )

    determ_df = pd.concat([irt_df, info_df])
    irt_line = (
        alt.Chart(format_df(determ_df))
        .mark_line()
        .encode(x=x_scale, y=alt.Y("value", title="", scale=y_scale), color=color_scale)
    )
    font_size = 18
    chart = (
        (random_band + random_line + irt_line)
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(
            labelFontSize=font_size, titleFontSize=font_size, symbolLimit=0, labelLimit=0,
        )
        .configure_header(labelFontSize=font_size)
        .configure(padding=0)
    )

    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "sampling_rank", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "sampling_rank", filetypes)


@register_plot("cat_sampling_stability")
def cat_sample_stability_plot(filetypes: List[str], commit: bool = False):
    input_dir = Path(conf["stability"]["cat_sampling"])
    random_df = pd.read_json(input_dir / "random_df.json")
    irt_df = pd.read_json(input_dir / "irt_df.json")
    info_df = pd.read_json(input_dir / "info_df.json")

    method_names = {
        "dev_high_disc_to_test": "High Discrimination",
        "dev_high_diff_to_test": "High Difficulty",
        "dev_high_disc_diff_to_test": "High Disc + Diff",
        "dev_info_to_test": "High Information",
        "dev_random_to_test": "Random",
    }

    def format_df(dataframe):
        return dataframe.assign(
            sampling_method=dataframe["variable"].map(lambda v: method_names[v])
        )

    x_scale = alt.X("trial_size", title="Development Set Sample Size", scale=alt.Scale(type="log"))
    y_scale = alt.Scale(zero=False)
    color_scale = alt.Color(
        "sampling_method",
        title="Sampling Method",
        legend=alt.Legend(orient="bottom-right", fillColor="white", padding=5, strokeColor="gray"),
        sort=[
            "High Information",
            "High Discrimination",
            "High Disc + Diff",
            "High Difficulty",
            "Random",
        ],
    )
    random_line = (
        alt.Chart(format_df(random_df))
        .mark_line()
        .encode(
            x=x_scale,
            y=alt.Y("mean(value)", scale=y_scale, title="Correlation to Test Rank"),
            color=color_scale,
        )
    )
    random_band = (
        alt.Chart(format_df(random_df))
        .mark_errorband(extent="ci")
        .encode(x=x_scale, y=alt.Y("value", title="", scale=y_scale), color=color_scale)
    )

    determ_df = pd.concat([irt_df, info_df])
    irt_line = (
        alt.Chart(format_df(determ_df))
        .mark_line()
        .encode(x=x_scale, y=alt.Y("value", title="", scale=y_scale), color=color_scale)
    )
    font_size = 18
    chart = (
        (random_band + random_line + irt_line)
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(
            labelFontSize=font_size, titleFontSize=font_size, symbolLimit=0, labelLimit=0,
        )
        .configure_header(labelFontSize=font_size)
        .configure(padding=0)
    )

    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "cat_sampling_rank", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "cat_sampling_rank", filetypes)


def label_experiment(label):
    if label.startswith("test_"):
        return "Dev Sample to Test"
    else:
        return "Dev Sample to Dev Sample"


def label_sig(fold: str):
    if fold == "dev":
        return "Dev Sample to Dev Sample"
    elif fold == "test":
        return "Dev Sample to Test"
    else:
        raise ValueError(f"Invalid fold: {fold}")


@functools.lru_cache()
def load_test_irt():
    test_irt_parsed = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["test"]["pyro"]["3PL"]["full"]) / "parameters.json"
    )
    test_preds = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"]["test"])
    mapping = read_json(conf["squad"]["dev_to_test"])
    dev_to_test = mapping["dev_to_test"]

    def get_test_irt(dev_id):
        if dev_id in dev_to_test:
            test_id = dev_to_test[dev_id]
            if test_id in test_irt_parsed.model_stats:
                return test_irt_parsed.model_stats[test_id].skill
            else:
                return None
        else:
            return None

    def get_test_classical(dev_id):
        if dev_id in dev_to_test:
            test_id = dev_to_test[dev_id]
            if test_id in test_preds.model_scores:
                return test_preds.model_scores[test_id]["exact_match"]
            else:
                return None
        else:
            return None

    return get_test_irt, get_test_classical


def rank_compute_bootstrap_ci(data_path: str, n_trials: int = 1000, fold: str = "dev"):
    """Given stability experiment, compute bootstrapped
    confidence intervals, and check if correlations are above 95%
    interval.

    Args:
        data_path (str): Path to dataframe stored in feather format with experiment
    """
    df = pd.read_feather(data_path)
    size = df["size"].iloc[0]
    trial_id = df["trial_id"].iloc[0]
    if fold == "test":
        get_test_irt, get_test_classical = load_test_irt()
        df["b_irt"] = df["subject_id"].map(get_test_irt)
        df["b_classical"] = df["subject_id"].map(get_test_classical)
        df = df.dropna(0)

    real_corr = df.corr(method="kendall")

    # Due to not implementing identifiability, IRT scores may be flipped
    # Detect that and adjust as necessary
    if real_corr["a_irt"].a_classical < 0:
        df["a_irt"] = -df["a_irt"]

    if real_corr["b_irt"].b_classical < 0:
        df["b_irt"] = -df["b_irt"]

    real_corr = df.corr(method="kendall")

    corr_diff = real_corr["a_irt"].b_irt - real_corr["a_classical"].b_classical
    a_classical_scores = df.a_classical.to_numpy()
    a_irt_scores = df.a_irt.to_numpy()
    indices = np.arange(0, len(a_classical_scores))
    # Build up a distribution of score differences
    diff_dist = []
    # Simulate a bunch of times
    n_subjects = len(a_classical_scores)

    for _ in range(n_trials):
        # Create a new similar DF, except sample with replacement one set of rankings
        # Be sure to keep pairs of irt/classical scores together
        sample_indices = np.random.choice(indices, n_subjects, replace=True)
        sample_classical = a_classical_scores[sample_indices]
        sample_irt = a_irt_scores[sample_indices]
        sample_df = pd.DataFrame(
            {
                "subject_id": df["subject_id"],
                # I'm not sure doing replacement is correct
                # Also not sure if n=161 is correct, seems odd,
                # but I'd be worried if I did only 20 that
                # the distribution of differences might be different
                "a_classical": sample_classical,
                "a_irt": sample_irt,
                # Keep one ranking the same
                "b_classical": df["b_classical"],
                "b_irt": df["b_irt"],
            }
        )
        sample_corr = sample_df.corr(method="kendall")

        # Grab correlations
        irt_corr = sample_corr.loc["a_irt"].b_irt
        classical_corr = sample_corr.loc["a_classical"].b_classical

        # Record the difference
        diff_dist.append(irt_corr - classical_corr)
    diff_df = pd.DataFrame({"diff": diff_dist})
    # Two tailed test, so divide by two
    alpha = 1 - 0.95

    lower, upper = diff_df["diff"].quantile([alpha, 1 - alpha])
    # significant = bool(corr_diff < lower or upper < corr_diff)
    significant = bool(upper < corr_diff)
    p_value = 1 - ((diff_df["diff"] < corr_diff).sum() / n_trials)
    return {
        "significant": significant,
        "p_value": float(p_value),
        "diff": float(corr_diff),
        "irt_corr": float(real_corr["a_irt"].b_irt),
        "classical_corr": float(real_corr["a_classical"].b_classical),
        "trial_size": int(size),
        "trial_id": int(trial_id),
        "lower": float(lower),
        "upper": float(upper),
        "alpha": alpha,
        "diff_dist": diff_dist,
    }


def process_trial_group(trial_size, trials):
    diff_dist = []
    for t in trials:
        diff_dist.extend(t["diff_dist"])
    diff_dist = np.array(diff_dist)
    for t in trials:
        p_value = 1 - (diff_dist < t["diff"]).mean()
        t["total_p_value"] = p_value
        yield t


def get_cached_rank_stability_sig(force: bool = False, n_trials: bool = 1000):
    input_dir = Path(conf["stability"]["ranking"])
    output_path = Path(conf["stability"]["ranking_sig"])
    if output_path.exists() and not force:
        console.log("Cached ranking stability found")
        return pd.read_feather(output_path)
    console.log("Cached ranking stability not found, computing...")

    console.log("Computing dev results")
    dev_results = (
        pseq(input_dir.glob("*.feather"))
        .map(lambda x: rank_compute_bootstrap_ci(x, n_trials=n_trials, fold="dev"))
        .list()
    )
    console.log("Computing test results")
    test_results = (
        pseq(input_dir.glob("*.feather"))
        .map(lambda x: rank_compute_bootstrap_ci(x, n_trials=n_trials, fold="test"))
        .list()
    )
    dev_processed = (
        seq(dev_results)
        .group_by(lambda x: x["trial_size"])
        .smap(process_trial_group)
        .flatten()
        .list()
    )
    test_processed = (
        seq(test_results)
        .group_by(lambda x: x["trial_size"])
        .smap(process_trial_group)
        .flatten()
        .list()
    )
    dev_df = pd.DataFrame(dev_processed).drop("diff_dist", axis=1)
    dev_df["fold"] = "dev"
    test_df = pd.DataFrame(test_processed).drop("diff_dist", axis=1)
    test_df["fold"] = "test"
    df = pd.concat([dev_df, test_df]).reset_index()
    df["experiment"] = df["fold"].map(label_sig)
    df.to_feather(output_path)
    return df


@register_plot("rank_sig")
def rank_stability_sig(filetypes: List[str], commit: bool = False, n_trials: int = 1000):
    df = get_cached_rank_stability_sig(force=False, n_trials=n_trials)
    font_size = 14
    chart = (
        # The plot gets too crowded if we include below 100, where
        # we have a higher density of experiments than 1 per 100 sizes
        alt.Chart(df[df["trial_size"] > 99])
        .mark_boxplot(size=5)
        .encode(
            x=alt.X("trial_size", title="Sample Size"),
            y=alt.Y("total_p_value", title="P-Value", axis=alt.Axis(tickCount=11),),
        )
        .properties(width=400, height=150)
        .facet(alt.Column("experiment", title=""))
        .configure_header(titleFontSize=font_size, labelFontSize=font_size)
    )

    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "ranking_stability_significance", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "ranking_stability_significance", filetypes)


@register_plot("stability")
def rank_stability_plot(filetypes: List[str], commit: bool = False):
    test_irt_parsed = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["test"]["pyro"]["3PL"]["full"]) / "parameters.json"
    )
    test_preds = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"]["test"])
    mapping = read_json(conf["squad"]["dev_to_test"])
    dev_to_test = mapping["dev_to_test"]
    df = create_rank_stability_df(
        dev_to_test=dev_to_test, test_preds=test_preds, test_irt_parsed=test_irt_parsed
    )

    names = {
        "abs_irt_corr": "IRT to IRT",
        "classical_corr": "Acc to Acc",
        "test_classical_sample_classical_corr": "Acc to Acc",
        "test_classical_sample_irt_corr": "IRT to Acc",
        "test_irt_sample_classical_corr": "Acc to IRT",
        "test_irt_sample_irt_corr": "IRT to IRT",
    }
    color_order = ["IRT to IRT", "Acc to Acc", "IRT to Acc", "Acc to IRT"]

    melt_df = df.drop(columns=["irt_corr"]).melt(id_vars=["trial_size", "trial_id"]).dropna(axis=0)
    excluded = ["IRT to Acc", "Acc to IRT"]
    console.log(melt_df.head())
    melt_df["correlation"] = melt_df["variable"].map(lambda v: names[v])
    melt_df = melt_df[melt_df["correlation"].map(lambda v: v not in excluded)]
    melt_df["experiment"] = melt_df["variable"].map(label_experiment)
    base = alt.Chart(melt_df).encode(
        x=alt.X(
            "trial_size",
            title="Development Set Sample Size",
            scale=alt.Scale(type="log", base=2, domain=[16, 6000]),
        ),
        color=alt.Color(
            "correlation",
            title="Correlation",
            scale=alt.Scale(scheme="category10"),
            sort=color_order,
            legend=alt.Legend(
                symbolOpacity=1,
                symbolType="circle",
                symbolStrokeWidth=3,
                orient="none",
                legendX=570,
                legendY=105,
                fillColor="white",
                strokeColor="gray",
                padding=5,
            ),
        ),
    )
    y_title = "Kendall Rank Correlation"
    line = base.mark_line(opacity=0.7).encode(
        y=alt.Y("mean(value):Q", scale=alt.Scale(zero=False), title=y_title),
    )
    band = base.mark_errorband(extent="ci").encode(
        y=alt.Y("value", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("correlation", sort=color_order),
    )
    font_size = 14
    chart = (
        (band + line)
        .properties(width=300, height=170)
        .facet(alt.Column("experiment", title=""))
        .configure_header(titleFontSize=font_size, labelFontSize=font_size)
        .resolve_axis(y="independent")
        .configure(padding=0)
    )
    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "stability_simulation_corr", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "stability_simulation_corr", filetypes)


def create_rank_stability_df(*, test_irt_parsed, dev_to_test, test_preds):
    def get_test_irt(dev_id):
        if dev_id in dev_to_test:
            test_id = dev_to_test[dev_id]
            if test_id in test_irt_parsed.model_stats:
                return test_irt_parsed.model_stats[test_id].skill
            else:
                return None
        else:
            return None

    def get_test_classical(dev_id):
        if dev_id in dev_to_test:
            test_id = dev_to_test[dev_id]
            if test_id in test_preds.model_scores:
                return test_preds.model_scores[test_id]["exact_match"]
            else:
                return None
        else:
            return None

    rows = []
    trials = {}
    input_dir = Path(conf["stability"]["ranking"])
    for path in input_dir.glob("*.feather"):
        exp_df = pd.read_feather(path)
        exp_df["abs_a_irt"] = exp_df["a_irt"].abs()
        exp_df["abs_b_irt"] = exp_df["b_irt"].abs()
        exp_df["test_classical"] = exp_df["subject_id"].map(get_test_classical)
        exp_df["test_irt"] = exp_df["subject_id"].map(get_test_irt)
        # Drop the rows missing test data
        exp_df = exp_df.dropna(0)
        size = exp_df.iloc[0]["size"]
        trial_id = exp_df.iloc[0].trial_id
        trials[(size, trial_id)] = exp_df
        corr = exp_df.corr(method="kendall")
        rows.append(
            {
                "trial_size": size,
                "trial_id": trial_id,
                "irt_corr": corr.loc["a_irt"].b_irt,
                "classical_corr": corr.loc["a_classical"].b_classical,
                "test_irt_sample_irt_corr": abs(corr.loc["test_irt"].a_irt),
                "test_irt_sample_classical_corr": abs(corr.loc["test_irt"].a_classical),
                "test_classical_sample_irt_corr": abs(corr.loc["test_classical"].a_irt),
                "test_classical_sample_classical_corr": abs(corr.loc["test_classical"].a_classical),
            }
        )
        rows.append(
            {
                "trial_size": size,
                "trial_id": trial_id,
                "irt_corr": None,
                "classical_corr": None,
                "test_irt_sample_irt_corr": abs(corr.loc["test_irt"].b_irt),
                "test_irt_sample_classical_corr": abs(corr.loc["test_irt"].b_classical),
                "test_classical_sample_irt_corr": abs(corr.loc["test_classical"].b_irt),
                "test_classical_sample_classical_corr": abs(corr.loc["test_classical"].b_classical),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_irt_corr"] = df["irt_corr"].abs()
    return df


# unregistering b/c the data doesn't seem to exist, and
# the commit flag wasn't there -- assuming it's an old plot?
# @register_plot("irt_correlation")
def irt_correlation_table(
    filetypes: List[str],
    include_multidim: bool = True,
    method: str = "kendall",
    commit: bool = False,
):
    irt_3pl = IrtParsed.from_irt_file("data/irt/squad/dev/pyro/3PL_full/parameters.json")
    irt_2pl = IrtParsed.from_irt_file("data/irt/squad/dev/pyro/2PL_full/parameters.json")
    irt_1pl = IrtParsed.from_irt_file("data/irt/squad/dev/pyro/1PL_full/parameters.json")
    models = [irt_3pl, irt_2pl, irt_1pl]
    if include_multidim:
        multidim = IrtParsed.from_multidim_1d_file(
            "data/multidim_irt/squad/dev/dim=1/models.jsonlines",
            "data/multidim_irt/squad/dev/dim=1/items.jsonlines",
        )
        models.append(multidim)
    subject_rows = []
    for sid in irt_3pl.model_ids:
        entry = {"subject_id": sid}
        for irt_model in models:
            entry[irt_model.irt_model.value] = irt_model.model_stats[sid].skill
        subject_rows.append(entry)
    subject_df = pd.DataFrame(subject_rows)

    console.log("Method:", method)
    subject_corr = subject_df.corr(method=method)
    console.log("Subject Correlations")
    console.log(subject_corr)

    names = {"pyro_3pl": "3PL", "pyro_2pl": "2PL", "pyro_1pl": "1PL"}
    print(
        subject_corr.applymap(lambda n: f"${to_precision(n, 3)}$")
        .rename(columns=names, index=names)
        .to_latex(column_format="l" + len(names) * "r", escape=False)
    )

    item_rows = []
    for item_id in irt_3pl.example_ids:
        entry = {"item_id": item_id}
        for irt_model in models:
            entry[irt_model.irt_model.value] = irt_model.example_stats[item_id].diff
        item_rows.append(entry)
    item_df = pd.DataFrame(item_rows)

    item_corr = item_df.corr(method=method)
    console.log("Item Correlations")
    console.log(item_corr)
    print(
        item_corr.applymap(lambda n: f"${to_precision(n, 3)}$")
        .rename(columns=names, index=names)
        .to_latex(column_format="l" + len(names) * "r", escape=False)
    )


def create_subject_df(*, dev_id_to_subject, dev_irt_params, dev_to_test, test_irt_params):
    subject_rows = []
    id_to_subject_stats = {}
    for subject_id in dev_id_to_subject.keys():
        subject = dev_id_to_subject[subject_id]
        entry = {
            "subject_id": subject_id,
            "name": subject["name"],
            "dev_em": subject["dev_em"],
            "test_em": subject["test_em"],
            "dev_skill": dev_irt_params.model_stats[subject_id].skill,
        }
        if subject_id in dev_to_test:
            test_subject_id = dev_to_test[subject_id]
            if test_subject_id in test_irt_params.model_stats:
                entry["test_skill"] = test_irt_params.model_stats[test_subject_id].skill
        subject_rows.append(entry)
        id_to_subject_stats[subject_id] = entry
    subject_df = pd.DataFrame(subject_rows)
    subject_df["overfit"] = subject_df["test_em"] - subject_df["dev_em"]
    return subject_df, id_to_subject_stats


def create_item_df(dev_irt_params):
    item_rows = []
    for item_id, item_stats in dev_irt_params.example_stats.items():
        item_rows.append({"item_id": item_id, "disc": item_stats.disc, "diff": item_stats.diff})
    all_item_df = pd.DataFrame(item_rows)
    item_df = all_item_df[all_item_df.disc >= 0.0]
    return item_df


order = ["Low", "Med-Low", "Med-High", "High"]


def categorize_disc(disc: float, disc_quantiles):
    for cat in order:
        if disc <= disc_quantiles[cat]:
            return cat
    return "High"


def categorize_diff(diff: float, diff_quantiles):
    for cat in order:
        if diff <= diff_quantiles[cat]:
            return cat
    return "High"


def create_confusion_df(*, dev_predictions, dev_irt_params, diff_quantiles, disc_quantiles):
    rows = []
    for subject_id, scored_predictions in dev_predictions.scored_predictions.items():
        for item_id, score in scored_predictions["exact_match"].items():
            item_stats = dev_irt_params.example_stats[item_id]
            rows.append(
                {
                    "item_id": item_id,
                    "subject_id": subject_id,
                    "response": score,
                    "diff": item_stats.diff,
                    "diff_cat": categorize_diff(item_stats.diff, diff_quantiles),
                    "disc": item_stats.disc,
                    "disc_cat": categorize_disc(item_stats.disc, disc_quantiles),
                }
            )
    all_df = pd.DataFrame(rows)
    all_df["n"] = 1
    df = all_df[all_df["disc"] >= 0]
    return df


@register_plot("confusion")
def confusion_plot(filetypes: List[str], commit: bool = False, irt_model: str = "3PL"):
    dev_irt_params = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["dev"]["pyro"][irt_model]["full"]) / "parameters.json"
    )
    dev_predictions = LeaderboardPredictions.parse_file(
        conf["squad"]["submission_predictions"]["dev"]
    )
    dev_id_to_subject = load_squad_submissions(dev_predictions)
    mapping = read_json(conf["squad"]["dev_to_test"])
    dev_to_test = mapping["dev_to_test"]
    test_irt_params = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["test"]["pyro"][irt_model]["full"]) / "parameters.json"
    )
    subject_df, id_to_subject_stats = create_subject_df(
        dev_id_to_subject=dev_id_to_subject,
        dev_irt_params=dev_irt_params,
        dev_to_test=dev_to_test,
        test_irt_params=test_irt_params,
    )
    item_df = create_item_df(dev_irt_params)

    diff_quantiles = {
        "Low": item_df["diff"].quantile(0.25),
        "Med-Low": item_df["diff"].quantile(0.5),
        "Med-High": item_df["diff"].quantile(0.75),
        "High": item_df["diff"].quantile(1),
    }
    disc_quantiles = {
        "Low": item_df["disc"].quantile(0.25),
        "Med-Low": item_df["disc"].quantile(0.5),
        "Med-High": item_df["disc"].quantile(0.75),
        "High": item_df["disc"].quantile(1),
    }
    df = create_confusion_df(
        dev_predictions=dev_predictions,
        dev_irt_params=dev_irt_params,
        diff_quantiles=diff_quantiles,
        disc_quantiles=disc_quantiles,
    )
    by_diff = df.groupby(["subject_id", "diff_cat"]).mean("n").reset_index()
    by_diff["category"] = by_diff["diff_cat"]
    by_diff["parameter"] = "Difficulty"

    by_disc = df.groupby(["subject_id", "disc_cat"]).mean("n").reset_index()
    by_disc["category"] = by_disc["disc_cat"]
    by_disc["parameter"] = "Discriminability"

    combined = pd.concat([by_diff, by_disc])

    def format_name(name):
        parens = name.split("(")
        return parens[0]

    combined["name"] = combined["subject_id"].map(
        lambda sid: format_name(id_to_subject_stats[sid]["name"])
    )
    combined["test_em"] = combined["subject_id"].map(
        lambda sid: id_to_subject_stats[sid]["test_em"]
    )
    combined["percent"] = combined["response"].map(lambda p: round(100 * p))

    selected_subject_ids = {
        # overfitting dev_em
        # "0x7a3431e690444afca4988f927ad23019": ["Overfit"],
        "0xa5265f8dbc424109a4573494c113235d": ["Overfit"],
        # top dev_skill
        # "0x8978eb3bd032447a80f27b2b82ad3b80": [
        #     "Top Dev Ability",
        #     "Top Test EM",
        # ],  # also top test_em
        # "0xe56a3accea374f9787255a85febd8497": ["Top Model"],
        # top  test_em
        # "0xc81e2e3395dd447eb85c899aa93d0d16": ["Top Model"],
        # "0x082fc49949b14c6aa3827bfebed5cc40": ["Top Model"],
        # top test skill
        "0x8978eb3bd032447a80f27b2b82ad3b80": ["Top Model"],
        # lowest dev_skill
        "0x8a3b01f4ded748df8b657684212372b4": ["Bottom Model"],
        # "0x2d5cf8f56e164de8837cb8ed30f15f59": ["Bottom Model"],
        # "0xfe18a19d54d44e2eaefd68836c3b388b": ["Hallmark"],
        # "0xfcd2efa17551478f96c593fe07eebd97": ["Hallmark"],
        "0xeb6fe173849a495b83eb4e56b172e02a": ["Hallmark"],
        "0x843f0d9f242f46b9803558614bff2f86": ["Hallmark"],
    }
    combined_filtered = combined[combined["subject_id"].map(lambda x: x in selected_subject_ids)]

    label_rows = []
    for sid, labels in selected_subject_ids.items():
        for l in labels:
            label_rows.append(
                {
                    "subject_id": sid,
                    "name": format_name(id_to_subject_stats[sid]["name"]),
                    "label": l,
                    "n": 1,
                    "test_em": id_to_subject_stats[sid]["test_em"],
                    "test_em_percent": round(id_to_subject_stats[sid]["test_em"]),
                }
            )
    label_df = pd.DataFrame(label_rows)
    subject_order = label_df.sort_values("test_em", ascending=False).name.tolist()

    label_chart = (
        alt.Chart(label_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "n", title="", axis=alt.Axis(labels=False, grid=False, ticks=False, domain=False),
            ),
            y=alt.Y(
                "name",
                title="Name",
                sort=subject_order,
                axis=alt.Axis(labels=True, ticks=True, orient="left"),
            ),
            color=alt.Color(
                "label",
                title="Description",
                legend=alt.Legend(
                    orient="left", offset=0,  # , direction="vertical", legendX=-250, legendY=50
                ),
                scale=alt.Scale(scheme="set2"),
            ),
        )
    )
    label_text = (
        alt.Chart(label_df)
        .mark_text(baseline="middle", dx=-12, color="black")
        .encode(
            x=alt.X("n", title="Test Acc", axis=alt.Axis(orient="top")),
            y=alt.Y("name", sort=subject_order),
            text=alt.Text("test_em_percent"),
        )
    )
    label_chart = (label_chart + label_text).properties(width=25)

    main_chart = (
        alt.Chart(combined_filtered)
        .mark_rect(stroke="black", strokeWidth=0.1)
        .encode(
            x=alt.X("category", title="", sort=order, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y(
                "name",
                title="",
                sort=subject_order,
                axis=alt.Axis(titlePadding=10, labels=False),
                scale=alt.Scale(paddingInner=0.2),
            ),
            color=alt.Color(
                "response",
                title="Dev Acc",
                scale=alt.Scale(scheme="magma"),
                legend=alt.Legend(offset=0),
            ),
        )
    )
    text = (
        alt.Chart(combined_filtered)
        .mark_text(baseline="middle")
        .encode(
            x=alt.X("category", sort=order),
            y=alt.Y("name", sort=subject_order),
            text=alt.Text("percent"),
            color=alt.condition(alt.datum.percent > 70, alt.value("black"), alt.value("white")),
        )
    )

    chart = main_chart + text
    hcat = alt.hconcat()
    for param in ["Difficulty", "Discriminability"]:
        hcat |= chart.transform_filter(datum.parameter == param).properties(title=param)
    chart = alt.hconcat(label_chart, hcat, spacing=0).configure(padding=0)

    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "irt_confusion", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "irt_confusion", filetypes)


@register_plot("irt_compare")
@requires_files(list(IRT_FILES.values()))
def plot_irt_comparison(filetypes: List[str], commit: bool = False):
    irt_reports = []
    for (model_type, eval_type), path in IRT_FILES.items():
        report = read_json(path)
        irt_reports.append(
            {
                "model": model_type,
                "evaluation": eval_type,
                "ROC AUC": report["roc_auc"],
                "Macro F1": report["classification_report"]["macro avg"]["f1-score"],
                # "Macro Precision": report["classification_report"]["macro avg"][
                #    "precision"
                # ],
                # "Macro Recall": report["classification_report"]["macro avg"]["recall"],
                #'weighted_f1': report['classification_report']['weighted avg']['f1-score'],
                #'weighted_precision': report['classification_report']['weighted avg']['precision'],
                #'weighted_recall': report['classification_report']['weighted avg']['recall'],
                "Accuracy": report["classification_report"]["accuracy"],
            }
        )
    report_df = pd.DataFrame(irt_reports)

    def to_precision_numbers(num, places):
        if isinstance(num, str):
            return r"\text{" + num + r"}"
        else:
            return to_precision(num, places)

    latex_out = (
        report_df[report_df.evaluation == "heldout"]
        .applymap(lambda n: f"${to_precision_numbers(n, 3)}$")
        .pivot(index="model", columns="evaluation")
        .reset_index()
        .to_latex(index=False, escape=False)
    )
    print(latex_out)

    df = report_df.melt(id_vars=["model", "evaluation"], var_name="metric")
    METRIC_SORT_ORDER = [
        "ROC AUC",
        "Macro F1",
        "Macro Precision",
        "Macro Recall",
        "Accuracy",
    ]
    heldout_df = df[df.evaluation == "heldout"]
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            color=alt.Color(
                "model",
                title="IRT Model",
                scale=alt.Scale(scheme="category10"),
                legend=alt.Legend(orient="top"),
            ),
            x=alt.X("model", title="", axis=alt.Axis(labels=False), sort=METRIC_SORT_ORDER,),
            y=alt.Y("value", title="Heldout Metric", scale=alt.Scale(zero=False, domain=[0.8, 1]),),
            tooltip="value",
        )
        .properties(width=100, height=150)
    )
    font_size = 18
    text = bars.mark_text(align="center", baseline="middle", dy=-7, fontSize=14).encode(
        text=alt.Text("value:Q", format=".2r"), color=alt.value("black")
    )

    chart = (
        alt.layer(bars, text, data=heldout_df)
        .facet(column=alt.Column("metric", title=""))
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_header(labelFontSize=font_size)
    )

    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "irt_model_comparison", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "irt_model_comparison", filetypes)


@register_plot("ablation")
@requires_files(list(ABLATION_FILES.values()))
def ablation_plot(filetypes: List[str], commit: bool = False):
    rows = []
    for (_, irt_type, feature_set), report_path in ABLATION_FILES.items():
        report = read_json(report_path)
        if feature_set in [
            "guids+stats",
            "guids+qwords",
            "topics_10",
            "topics_50",
            "topics_500",
            "topics_100",
            "LM - Title",
        ]:
            continue

        if feature_set == "irt":
            feature_set = "IRT"
        elif feature_set == "guids":
            feature_set = "Subj & Item ID"
        elif feature_set == "ex_id":
            feature_set = "Item ID"
        elif feature_set == "m_id":
            feature_set = "Subject ID"
        elif feature_set == "qwords":
            feature_set = "Question"
        elif feature_set == "cwords":
            feature_set = "Context"
        elif feature_set == "topics_1000":
            feature_set = "Topics 1K"
        elif feature_set == "topics_100":
            feature_set = "Topics 100"
        else:
            feature_set = feature_set.capitalize()

        if feature_set == "All":
            name = "LM All"
        else:
            name = f"LM +{feature_set}"
        rows.append(
            {
                "features": name,
                "irt": irt_type,
                "ROC AUC": report["roc_auc"],
                "Macro F1": report["classification_report"]["macro avg"]["f1-score"],
                # "Macro Precision": report["classification_report"]["macro avg"][
                #     "precision"
                # ],
                # "Macro Recall": report["classification_report"]["macro avg"]["recall"],
                #'weighted_f1': report['classification_report']['weighted avg']['f1-score'],
                #'weighted_precision': report['classification_report']['weighted avg']['precision'],
                #'weighted_recall': report['classification_report']['weighted avg']['recall'],
                "Accuracy": report["classification_report"]["accuracy"],
            }
        )

    df = pd.DataFrame(rows).melt(id_vars=["features", "irt"], var_name="metric")

    IRT_FILES[("multidim", "heldout")] = "data/irt/squad/dev/pyro/multidim_10d_heldout/report.json"
    irt_reports = []
    for (model_type, eval_type), path in IRT_FILES.items():
        report = read_json(path)
        if model_type == "1PL":
            model_type = "Base"
        elif model_type == "2PL":
            model_type = "Disc"
        elif model_type == "3PL":
            model_type = "Feas"
        elif model_type == "multidim":
            model_type = "Vec"
        irt_reports.append(
            {
                "features": f"IRT-{model_type}",
                "irt": model_type,
                "evaluation": eval_type,
                "ROC AUC": report["roc_auc"],
                "Macro F1": report["classification_report"]["macro avg"]["f1-score"],
                # "Macro Precision": report["classification_report"]["macro avg"][
                #    "precision"
                # ],
                # "Macro Recall": report["classification_report"]["macro avg"]["recall"],
                #'weighted_f1': report['classification_report']['weighted avg']['f1-score'],
                #'weighted_precision': report['classification_report']['weighted avg']['precision'],
                #'weighted_recall': report['classification_report']['weighted avg']['recall'],
                "Accuracy": report["classification_report"]["accuracy"],
            }
        )
    report_df = pd.DataFrame(irt_reports)
    report_df = report_df[report_df["evaluation"] == "heldout"]
    report_df = report_df.drop(["evaluation"], axis=1)
    report_df = report_df.melt(id_vars=["features", "irt"], var_name="metric")

    df = pd.concat([report_df, df])
    # cutting the following chart types:
    #  - guid+stats
    #  - guids+qwords
    #  - topics_10
    #  - topics_50
    #  - topics_500

    # to_remove = ["guids+stats", "guids+qwords", "topics_10", "topics_50", "topics_500"]
    # to_remove = [f"LM - {name}" for name in to_remove]
    # df = df[~df["features"].isin(to_remove)]

    group_sizes = df.groupby(["features", "irt", "metric"]).count().value.unique()
    if len(group_sizes) != 1 or 1 not in group_sizes:
        raise ValueError(f"Bad group sizes: {group_sizes}")

    METRIC_SORT_ORDER = [
        "ROC AUC",
        "Macro F1",
        "Macro Precision",
        "Macro Recall",
        "Accuracy",
    ]

    def sort_order(val):
        return METRIC_SORT_ORDER.index(val)

    label_order = [
        "IRT-Vec",
        "IRT-Feas",
        "IRT-Disc",
        "IRT-Base",
        "LM All",
        "LM +IRT",
        "LM +Subj & Item IDs",
        "LM +Item ID",
        "LM +Subject ID",
        "LM +Question",
        "LM +Context",
        "LM +Stats",
    ]
    base = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            color=alt.Color(
                "features",
                title="Features",
                scale=alt.Scale(scheme="category20"),
                sort=label_order,
                legend=alt.Legend(symbolLimit=0, labelLimit=0),
            )
        )
    )

    chart = alt.hconcat()
    first = True
    metric_names = sorted(df.metric.unique(), key=sort_order)
    for metric in metric_names:
        # if first:
        #     title = "Metric Value"
        #     first = False
        # else:
        #     title = ""
        new_chart = (
            base.encode(
                x=alt.X(
                    "features", title="", axis=alt.Axis(labelAngle=-45, labels=False), sort="-y",
                ),
                y=alt.Y("value", title="", scale=alt.Scale(domain=(0, 1))),
            )
            .transform_filter(datum.metric == metric)
            .properties(title=metric)  # , height=100, width=50)
        )
        chart |= new_chart
    font_size = 25
    chart = (
        chart.configure_legend(columns=2, labelFontSize=font_size, titleFontSize=font_size)
        .configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_header(labelFontSize=font_size, titleFontSize=font_size)
        .configure_title(fontSize=font_size)
        # .configure(padding=0)
    )
    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "vw_ablation", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "vw_ablation", filetypes)


def load_predictions():
    squad = load_squad_v2(conf["squad"]["dev_v2"])
    qid_to_title = {}
    qid_to_is_impossible = {}
    for page in squad.data:
        for paragraph in page.paragraphs:
            for q in paragraph.qas:
                qid_to_title[q.id] = page.title
                qid_to_is_impossible[q.id] = q.is_impossible
    rows = []
    for line in read_jsonlines(conf["squad"]["leaderboard"]["dev"], lazy=True):
        submission_id = line["submission_id"]
        for pred in line["predictions"].values():
            example_id = pred["example_id"]
            em_score = pred["scores"]["exact_match"]
            rows.append(
                {
                    "submission_id": submission_id,
                    "example_id": example_id,
                    "em_score": em_score,
                    "is_impossible": "Unanswerable"
                    if qid_to_is_impossible[example_id]
                    else "Answerable",
                    "title": qid_to_title[example_id],
                }
            )
    return pd.DataFrame(rows)


@register_plot("question_dist")
def question_dist_plot(filetypes: List[str], commit: bool = False):
    df = load_predictions()
    acc_df = df.groupby(["example_id", "is_impossible", "title"]).mean().reset_index()
    histogram = (
        alt.Chart(acc_df)
        .mark_bar()
        .encode(
            x=alt.X("em_score", bin=alt.Bin(maxbins=100), title="SQuAD Exact Match Score"),
            y=alt.Y("count()", title="Number of Questions"),
        )
    )
    save_chart(histogram, PAPERS_ROOT / "auto_fig" / "question_acc_hist", filetypes)

    yscale = alt.Scale(domain=(0, 1))
    boxplot = alt.vconcat()
    for is_impossible in ("Answerable", "Unanswerable"):
        if is_impossible == "Answerable":
            boxplot_title = ""
            wiki_labels = False
        else:
            boxplot_title = "Wikipedia Page"
            wiki_labels = True

        is_impossible_boxplot = (
            alt.Chart(acc_df)
            .mark_boxplot()
            .encode(
                x=alt.X(
                    "title", title=boxplot_title, axis=alt.Axis(labelAngle=-45, labels=wiki_labels),
                ),
                y=alt.Y("em_score", title="SQuAD Exact Match Score", scale=yscale),
                color=alt.Color(
                    "title",
                    title="Wiki Page",
                    legend=alt.Legend(symbolLimit=40),
                    scale=alt.Scale(scheme="category20"),
                ),
            )
        ).transform_filter(datum.is_impossible == is_impossible)

        is_impossible_histogram = (
            alt.Chart(acc_df)
            .mark_area(interpolate="step")
            .encode(
                x=alt.X("count()", title="Count", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(
                    "em_score",
                    bin=alt.Bin(maxbins=50),
                    scale=yscale,
                    title="",
                    axis=alt.Axis(values=np.linspace(0, 1, 11)),
                ),
            )
            .properties(width=60)
            .transform_filter(datum.is_impossible == is_impossible)
        )
        boxplot &= is_impossible_boxplot | is_impossible_histogram
    if commit:
        save_chart(boxplot, COMMIT_AUTO_FIGS / "question_acc_boxplot", filetypes)
    else:
        save_chart(boxplot, AUTO_FIG / "question_acc_boxplot", filetypes)


def create_irt_dist_chart(irt_results: IrtParsed):
    rows = []
    for example in irt_results.example_stats.values():
        rows.append(
            {
                "disc": example.disc,
                "diff": example.diff,
                "lambda": example.lambda_,
                "item_id": example.example_id,
            }
        )

    def assign_feas_bin(feas):
        if feas < 0.33:
            return "Low"
        elif feas < 0.66:
            return "Mid"
        else:
            return "High"

    df = pd.DataFrame(rows)
    df["feas_bin"] = df["lambda"].map(assign_feas_bin)
    diff_min = np.floor(df["diff"].min())
    diff_max = np.ceil(df["diff"].max())
    diff_scale = alt.Scale(domain=(diff_min, diff_max))

    disc_min = np.floor(df["disc"].min())
    disc_max = np.ceil(df["disc"].max())
    disc_scale = alt.Scale(domain=(disc_min, disc_max))

    ratio = 1.5
    points = (
        alt.Chart(df)
        .mark_point(filled=True)
        .encode(
            x=alt.X("diff", title="Difficulty (𝜃)", scale=diff_scale),
            y=alt.Y("disc", title="Discriminability (𝛾)", scale=disc_scale),
            color=alt.Color(
                "lambda",
                title="Feasibility (λ)",
                scale=alt.Scale(scheme="redyellowblue"),
                legend=alt.Legend(
                    direction="horizontal",
                    orient="none",
                    legendX=240,
                    legendY=0,
                    gradientLength=80,
                ),
            ),
            size=alt.value(3),
            tooltip=alt.Tooltip(["item_id", "diff", "disc", "lambda"]),
        )
    ).properties(width=ratio * BASE_SIZE, height=ratio * BASE_SIZE)
    top_hist = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("diff", bin=alt.Bin(maxbins=50), stack=None, title="", scale=diff_scale),
            y=alt.Y("count()", stack=True, title=""),
        )
        .properties(height=40, width=ratio * BASE_SIZE)
    )
    right_hist = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("count()", stack=True, title=""),
            y=alt.Y("disc", bin=alt.Bin(maxbins=50), stack=None, title="", scale=disc_scale),
        )
    ).properties(width=40, height=ratio * BASE_SIZE)
    annotations = (
        alt.Chart(pd.DataFrame([{"text": "Annotation|Error", "x": -4, "y": -6}]))
        .mark_text(lineBreak="|", align="center")
        .encode(x="x", y="y", text="text")
    )
    # points = points + annotations
    chart = top_hist & (points | right_hist)
    chart = chart.configure_concat(spacing=10)

    base = alt.Chart(df)
    base = (
        alt.Chart(df)
        .transform_joinaggregate(total="count(*)")
        .transform_calculate(pct="1 / datum.total")
        .encode(
            x=alt.X("lambda", title="Probability of Feasibility (λ)", bin=alt.Bin(maxbins=49),),
        )
    )
    counts = base.mark_bar().encode(
        y=alt.Y(
            "count()", title="Count", scale=alt.Scale(type="log"), axis=alt.Axis(orient="left"),
        )
    )
    pcts = base.mark_bar().encode(
        y=alt.Y(
            "sum(pct):Q",
            title="Percentage",
            scale=alt.Scale(type="log"),
            axis=alt.Axis(orient="right", format="%"),
        )
    )
    lambda_dist_chart = alt.layer(counts, pcts).resolve_scale(y="independent")
    return chart, lambda_dist_chart


@register_plot("irt_dist")
@requires_file(PAPERS_ROOT / "auto_data" / "data/irt/squad/dev/pyro/3PL_full/parameters.json")
def plot_irt_dist(filetypes: List[str], commit: bool = False):
    irt_results = IrtParsed.from_irt_file(
        PAPERS_ROOT / "auto_data/data/irt/squad/dev/pyro/3PL_full/parameters.json"
    )
    chart, lambda_dist_chart = create_irt_dist_chart(irt_results)

    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "irt_example_dist", filetypes)
        save_chart(lambda_dist_chart, COMMIT_AUTO_FIGS / "irt_lambda_dist", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "irt_example_dist", filetypes)
        save_chart(lambda_dist_chart, AUTO_FIG / "irt_lambda_dist", filetypes)


@register_plot("irt_acc")
def plot_irt_acc(filetypes: List[str], commit: bool = False):
    base_parsed_irt = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["dev"]["pyro"]["3PL"]["full"]) / "parameters.json"
    )
    example_ids = base_parsed_irt.example_ids

    data = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"]["dev"])
    example_accuracy = defaultdict(float)
    for model_scores in data.scored_predictions.values():
        for ex_id in example_ids:
            example_accuracy[ex_id] += model_scores["exact_match"][ex_id]
    n_models = len(data.scored_predictions)
    for ex_id, correct in example_accuracy.items():
        example_accuracy[ex_id] = correct / n_models
    rows = []
    for irt_type in ("1PL", "2PL", "3PL"):
        parsed_irt = IrtParsed.from_irt_file(
            Path(conf["irt"]["squad"]["dev"]["pyro"][irt_type]["full"]) / "parameters.json"
        )

        for ex_id in example_ids:
            stats = parsed_irt.example_stats[ex_id]
            rows.append(
                {
                    "disc": stats.disc,
                    "diff": stats.diff,
                    "acc": example_accuracy[ex_id],
                    "irt": irt_type,
                }
            )
    df = pd.DataFrame(rows)
    chart = alt.hconcat()

    # for irt_type in ("1PL", "2PL", "3PL"):
    for irt_type in ("2PL",):
        if irt_type == "1PL":
            scatter = (
                alt.Chart(df)
                .mark_point()
                .encode(x=alt.X("acc", title="Accuracy"), y=alt.Y("diff", title="IRT Difficulty"),)
            )
        else:
            scatter = (
                alt.Chart(df)
                .mark_point()
                .encode(
                    x=alt.X("acc", title="Accuracy"),
                    y=alt.Y("diff", title="IRT Difficulty"),
                    color=alt.Color(
                        "disc",
                        title="IRT Discriminability",
                        scale=alt.Scale(scheme="cividis"),
                        legend=alt.Legend(
                            # direction="horizontal",
                            orient="top",
                            # legendX=930,
                            # legendY=10,
                            fillColor="white",
                        ),
                    ),
                )
            )
        if irt_type == "1PL":
            title = "1PL (No Discriminability Parameter)"
        else:
            title = irt_type
        scatter = scatter.transform_filter(datum.irt == irt_type).properties(
            title=f"IRT Model: {title}",
        )
        chart |= scatter
    chart = chart.resolve_scale(color="independent")
    if commit:
        save_chart(chart, COMMIT_AUTO_FIGS / "irt_acc_dist", filetypes)
    else:
        save_chart(chart, AUTO_FIG / "irt_acc_dist", filetypes)


def main(
    plot: Optional[str] = None,
    commit: bool = False,
    filetypes: List[str] = typer.Option(["pdf", "png", "svg", "json"]),
):
    console.log("Registered plots:", PLOTS)
    if plot is None:
        for name, func in PLOTS.items():
            console.log("Running: ", name)
            func(filetypes, commit=commit)
    else:
        PLOTS[plot](filetypes, commit=commit)
