"""Copyright (c) Facebook, Inc. and its affiliates."""
import itertools
import json
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi_cache import caches, close_caches
from fastapi_cache.backends.memory import CACHE_KEY, InMemoryCacheBackend
from pedroai.io import read_json
from pydantic import BaseModel

from leaderboard import stats
from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import (
    ExampleStats,
    IrtModelType,
    IrtResults,
    LeaderboardPredictions,
    LeaderboardSubmissions,
    ModelStats,
)
from leaderboard.log import get_logger
from leaderboard.www.database import (
    Example,
    SessionLocal,
    StatisticalTest,
    Submission,
    get_db,
)

alt.data_transformers.disable_max_rows()

log = get_logger(__name__)


def app_cache():
    return caches.get(CACHE_KEY)


class Cache(BaseModel):
    stan_1pl: Optional[IrtResults]
    pyro_1pl: Optional[IrtResults]
    stan_2pl: Optional[IrtResults]
    pyro_2pl: Optional[IrtResults]

    stan_1pl_map: Optional[Dict[str, ExampleStats]]
    pyro_1pl_map: Optional[Dict[str, ExampleStats]]
    stan_2pl_map: Optional[Dict[str, ExampleStats]]
    pyro_2pl_map: Optional[Dict[str, ExampleStats]]

    stan_2pl_skill: Optional[Dict[str, ModelStats]]
    pyro_2pl_skill: Optional[Dict[str, ModelStats]]

    metadata: Optional[LeaderboardSubmissions]
    predictions: Optional[LeaderboardPredictions]
    id_to_submission: Optional[Dict]
    model_df: Optional[pd.DataFrame]
    example_df: Optional[pd.DataFrame]
    stats_df: Optional[pd.DataFrame]
    squad_dataset: Optional[Dict]
    squad_lookup: Optional[Dict]
    qid_to_article: Optional[Dict]

    initialized: bool

    class Config:
        arbitrary_types_allowed = True


def chunks(lst: List, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


api_app = FastAPI()
data_cache = Cache(initialized=False)


def to_cache_key(payload: Dict):
    return json.dumps(payload, sort_keys=True)


def create_model_df():
    model_rows = []
    for model_id, scores in data_cache.predictions.model_scores.items():
        submission = data_cache.id_to_submission[str(model_id)]
        for metric_name, metric_value in scores.items():
            model_rows.append(
                {
                    "metric": metric_name,
                    "score": metric_value,
                    "model_id": submission.submit_id,
                    "name": submission.name,
                }
            )
    for name, results in [
        ("stan_ability", data_cache.stan_1pl),
        ("pyro_ability", data_cache.pyro_1pl),
    ]:
        for (model_id, submit_id), score in zip(results.model_ids.items(), results.ability):
            submission = data_cache.id_to_submission[str(submit_id)]
            model_rows.append(
                {
                    "metric": name,
                    "score": score,
                    "model_id": submission.submit_id,
                    "name": submission.name,
                }
            )

    return pd.DataFrame(model_rows)


def create_example_df():
    example_rows = []
    for name, results in [("stan", data_cache.stan_1pl), ("pyro", data_cache.pyro_1pl)]:
        for example_id, diff in zip(results.example_ids, results.diff):
            example_rows.append({"irt": name, "difficulty": diff, "example_id": example_id})
    example_df = pd.DataFrame(example_rows)
    return example_df


def create_stats_tests_df():
    results = []
    for t in stats.TESTS:
        results.extend(read_json(DATA_ROOT / f"data/stats/full/{t}.json")["results"])
    df = pd.DataFrame(results)
    df_sample = df[df.pvalue > 0.05].sample(600)
    return df_sample


def load_squad_full():
    dataset = read_json(DATA_ROOT / conf["squad"]["dev_v2"])["data"]
    question_map = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for question in p["qas"]:
                question["article"] = article["title"]
                question_map[question["id"]] = question
    return dataset, question_map


def load_irt(fold: str = "dev"):
    data_cache.stan_1pl = IrtResults.parse_file(
        DATA_ROOT / conf["irt"]["squad"][fold]["stan"]["1PL"]["full"] / "parameters.json"
    )
    data_cache.pyro_1pl = IrtResults.parse_file(
        DATA_ROOT / conf["irt"]["squad"][fold]["pyro"]["1PL"]["full"] / "parameters.json"
    )
    data_cache.stan_2pl = IrtResults.parse_file(
        DATA_ROOT / conf["irt"]["squad"][fold]["stan"]["2PL"]["full"] / "parameters.json"
    )
    data_cache.pyro_2pl = IrtResults.parse_file(
        DATA_ROOT / conf["irt"]["squad"][fold]["pyro"]["2PL"]["full"] / "parameters.json"
    )

    data_cache.stan_2pl_map = {}
    n_stan_2pl_examples = len(data_cache.stan_2pl.example_ids)
    for str_idx, example_id in data_cache.stan_2pl.example_ids.items():
        # STAN indexing is one based, so convert to zero based
        idx = int(str_idx) - 1
        if idx < 0 or idx > n_stan_2pl_examples - 1:
            raise ValueError(f"Invalid index: {idx}")
        data_cache.stan_2pl_map[example_id] = ExampleStats(
            irt_model=IrtModelType.stan_2pl.value,
            example_id=example_id,
            diff=data_cache.stan_2pl.diff[idx],
            disc=data_cache.stan_2pl.disc[idx],
        )

    data_cache.pyro_2pl_map = {}
    n_pyro_2pl_examples = len(data_cache.pyro_2pl.example_ids)
    for str_idx, example_id in data_cache.pyro_2pl.example_ids.items():
        idx = int(str_idx)
        if idx < 0 or idx > n_pyro_2pl_examples - 1:
            raise ValueError(f"Invalid index: {idx}")
        data_cache.pyro_2pl_map[example_id] = ExampleStats(
            irt_model=IrtModelType.pyro_2pl.value,
            example_id=example_id,
            diff=data_cache.pyro_2pl.diff[idx],
            disc=data_cache.pyro_2pl.disc[idx],
        )

    data_cache.stan_2pl_skill = {}
    n_stan_2pl_models = len(data_cache.stan_2pl.model_ids)
    for str_idx, model_id in data_cache.stan_2pl.model_ids.items():
        idx = int(str_idx) - 1
        if idx < 0 or idx > n_stan_2pl_models - 1:
            raise ValueError(f"Invalid index: {idx}")
        data_cache.stan_2pl_skill[model_id] = ModelStats(
            irt_model=IrtModelType.stan_2pl.value,
            model_id=model_id,
            skill=data_cache.stan_2pl.ability[idx],
        )

    data_cache.pyro_2pl_skill = {}
    n_pyro_2pl_models = len(data_cache.pyro_2pl.model_ids)
    for str_idx, model_id in data_cache.pyro_2pl.model_ids.items():
        idx = int(str_idx)
        if idx < 0 or idx > n_pyro_2pl_models - 1:
            raise ValueError(f"Invalid index: {idx}")
        data_cache.pyro_2pl_skill[model_id] = ModelStats(
            irt_model=IrtModelType.pyro_2pl.value,
            model_id=model_id,
            skill=data_cache.pyro_2pl.ability[idx],
        )


def initialize_cache():
    if not data_cache.initialized:
        load_irt()
        data_cache.metadata = LeaderboardSubmissions.parse_file(
            DATA_ROOT / conf["squad"]["submission_metadata"]
        )
        data_cache.predictions = LeaderboardPredictions.parse_file(
            DATA_ROOT / conf["squad"]["submission_predictions"]["dev"]
        )
        data_cache.id_to_submission = {s.submit_id: s for s in data_cache.metadata.submissions}
        data_cache.model_df = create_model_df()
        data_cache.example_df = create_example_df()
        data_cache.stats_df = create_stats_tests_df()
        data_cache.squad_dataset, data_cache.squad_lookup = load_squad_full()
        data_cache.qid_to_article = {}
        for article in data_cache.squad_dataset:
            for par in article["paragraphs"]:
                for question in par["qas"]:
                    data_cache.qid_to_article[question["id"]] = article["title"].replace("_", " ")
        data_cache.initialized = True
        log.info("Data cache initialized")


@api_app.get("/submissions")
async def read_submissions(
    db: SessionLocal = Depends(get_db), cache: InMemoryCacheBackend = Depends(app_cache)
):
    in_cache = await cache.get("/submissions")
    if in_cache is not None:
        log.info("Loaded from cache: /submissions")
        return in_cache

    submissions = []
    for db_sub in db.query(Submission):
        sub = db_sub.to_dict()
        sub["dev_skill"] = data_cache.pyro_2pl_skill[db_sub.submission_id].skill
        if sub["dev_scores"]["exact_match"] is not None:
            submissions.append(sub)
    submissions = sorted(submissions, key=lambda x: x["dev_scores"]["exact_match"], reverse=True,)
    response = {"submissions": submissions}
    await cache.set("/submissions", response)
    return response


def rename_metric(metric: str):
    if metric == "exact_match":
        return "Exact Match"
    elif metric == "f1":
        return "F1"
    else:
        return metric


@api_app.get("/submissions/plot")
async def plot_submissions(
    db: SessionLocal = Depends(get_db), cache: InMemoryCacheBackend = Depends(app_cache)
):
    in_cache = await cache.get("/submissions/plot")
    if in_cache is not None:
        log.info("Loaded from cache: /submissions/plot")
        return in_cache

    submissions = db.query(Submission)
    rows = []
    for sub in submissions:
        for metric, value in sub.dev_scores.items():
            rows.append(
                {
                    "metric": rename_metric(metric),
                    "value": 100 * value,
                    "submission_id": sub.submission_id,
                    "name": sub.name,
                    "created": sub.created,
                    "fold": "dev",
                }
            )
        for metric, value in sub.test_scores.items():
            rows.append(
                {
                    "metric": rename_metric(metric),
                    "value": value,
                    "submission_id": sub.submission_id,
                    "name": sub.name,
                    "created": sub.created,
                    "fold": "test",
                }
            )
    df = pd.DataFrame(rows)
    legend_selection = alt.selection_multi(fields=["metric"], bind="legend")
    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X("created", title="Submission Date"),
            y=alt.Y("value", title="Metric Value"),
            color=alt.Color("metric", title="Metric"),
            tooltip=["name", "metric", "value"],
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.2)),
        )
        .add_selection(legend_selection)
        .facet("fold", title="SQuAD 2.0 Leaderboard Timeline", columns=2)
        .interactive()
    )
    response = chart.to_dict()
    await cache.set("/submissions/plot", response)
    return response


@api_app.get("/stats/plot")
async def plot_stats(cache: InMemoryCacheBackend = Depends(app_cache)):
    in_cache = await cache.get("/stats/plot")
    if in_cache is not None:
        log.info("Loaded from cache: /stats/plot")
        return in_cache

    base = alt.Chart(data_cache.stats_df).interactive()
    points = (
        base.mark_point()
        .encode(
            x=alt.X("diff", title="Exact Match Score Difference"),
            y=alt.Y("pvalue", title="P-Value"),
            shape=alt.Shape("test:N", title="Statistical Test"),
            color=alt.Color("max_score", title="Max Score"),
        )
        .properties(width=600, height=600)
    )
    tick_axis = alt.Axis(labels=False, domain=False, ticks=False)
    x_ticks = (
        base.mark_tick()
        .encode(
            alt.X("diff", axis=tick_axis, title="Exact Match Score Difference"),
            alt.Y("test", title="", axis=tick_axis),
            color=alt.Color("test", title="Statistical Test"),
        )
        .properties(width=600)
    )

    y_ticks = (
        base.mark_tick()
        .encode(
            alt.X("test", title="", axis=tick_axis),
            alt.Y("pvalue", axis=tick_axis, title="P-Value"),
            color=alt.Color("test", title="Statistical Test"),
        )
        .properties(height=600)
    )

    # Build the chart
    chart = y_ticks | (points & x_ticks)
    response = chart.to_dict()
    await cache.set("/stats/plot", response)
    return response


class StatsByModelRequest(BaseModel):
    submission_id: str


@api_app.post("/stats/by-model")
async def stats_by_model(
    request: StatsByModelRequest,
    cache: InMemoryCacheBackend = Depends(app_cache),
    db: SessionLocal = Depends(get_db),
):
    key = to_cache_key({"path": "/stats/by-model", "submission_id": request.submission_id})
    in_cache = await cache.get(key)
    if in_cache is not None:
        log.info("Loaded from cache: /stats/by-model")
        return in_cache

    tests = db.query(StatisticalTest).filter(
        (StatisticalTest.model_a_id == request.submission_id)
        | (StatisticalTest.model_b_id == request.submission_id)
    )
    response = {"tests": [t.to_dict() for t in tests]}
    await cache.set(key, response)
    return response


class PairwiseRequest(BaseModel):
    submission_id_1: str
    submission_name_1: str
    submission_id_2: str
    submission_name_2: str


def score_to_name(score: float):
    if score == 1:
        return "Correct"
    elif score == 0:
        return "Incorrect"
    else:
        raise ValueError(f"Invalid Score: {score}")


@api_app.post("/pairwise/plot")
async def pairwise_plot(
    pairwise_request: PairwiseRequest, cache: InMemoryCacheBackend = Depends(app_cache)
):
    sub_id_1 = pairwise_request.submission_id_1
    sub_id_2 = pairwise_request.submission_id_2
    key = to_cache_key({"path": "/pairwise/plot", "sub_id_1": sub_id_1, "sub_id_2": sub_id_2})
    in_cache = await cache.get(key)
    if in_cache is not None:
        log.info("Loaded from cache: /pairwise/plot")
        return in_cache

    rows = []
    scored_preds_1 = data_cache.predictions.scored_predictions[sub_id_1]["exact_match"]
    scored_preds_2 = data_cache.predictions.scored_predictions[sub_id_2]["exact_match"]
    for qid in data_cache.squad_lookup.keys():
        score_1 = scored_preds_1[qid]
        score_2 = scored_preds_2[qid]
        article = data_cache.qid_to_article[qid]
        rows.append(
            {
                "sub_id_1": sub_id_1,
                "sub_id_2": sub_id_2,
                "score_1": score_to_name(score_1),
                "score_2": score_to_name(score_2),
                "metric": "exact_match",
                "qid": qid,
                "fold": "dev",
                "article": article,
            }
        )
    df = pd.DataFrame(rows)
    selection = alt.selection_multi(fields=["article"], bind="legend")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count()", title="Number of Questions"),
            y=alt.Y("article", title="Wikipedia Article"),
            color=alt.Color("article", title="Wikipedia Article"),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            row=alt.Row(
                "score_1", title=f"Model 1 Correct/Incorrect: {pairwise_request.submission_name_1}",
            ),
            column=alt.Column(
                "score_2", title=f"Model 2 Correct/Incorrect: {pairwise_request.submission_name_2}",
            ),
        )
        .add_selection(selection)
        .properties(width=320, height=350)
        .resolve_scale(x="independent")
    )
    response = chart.to_dict()
    await cache.set(key, response)
    return response


class PlotSubmissionsRequest(BaseModel):
    submission_ids: List[str]


@api_app.post("/submissions/plot_compare")
async def plot_compare(
    compare_request: PlotSubmissionsRequest,
    db: SessionLocal = Depends(get_db),
    cache: InMemoryCacheBackend = Depends(app_cache),
):
    key = to_cache_key({"path": "/submissions/plot_compare", "payload": compare_request.dict()})
    in_cache = await cache.get(key)
    if in_cache is not None:
        log.info("Loaded from cache: /submissions/plot_compare")
        return in_cache

    submissions = db.query(Submission).filter(
        Submission.submission_id.in_(compare_request.submission_ids)
    )
    rows = []
    for sub in submissions:
        for metric, value in sub.dev_scores.items():
            rows.append(
                {
                    "metric": rename_metric(metric),
                    "value": 100 * value,
                    "submission_id": sub.submission_id,
                    "name": sub.name,
                    "fold": "dev",
                }
            )
        for metric, value in sub.test_scores.items():
            rows.append(
                {
                    "metric": rename_metric(metric),
                    "value": value,
                    "submission_id": sub.submission_id,
                    "name": sub.name,
                    "fold": "test",
                }
            )
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value", title="Metric Value"),
            y=alt.Y("name", title="Model"),
            color=alt.Color("name"),
            row="metric",
            column="fold",
        )
        .properties(title="Model Comparison")
        .interactive()
    )
    response = chart.to_dict()
    await cache.set(key, response)
    return response


@api_app.get("/examples")
async def read_examples(
    db: SessionLocal = Depends(get_db), cache: InMemoryCacheBackend = Depends(app_cache)
):
    in_cache = await cache.get("/examples")
    if in_cache is not None:
        log.info("Loaded from cache: /examples")
        return in_cache

    examples = db.query(Example)
    response = {
        "examples": [
            {
                "example_id": e.example_id,
                "task": e.task,
                "data": data_cache.squad_lookup[e.example_id],
                "disc_pyro_2pl": data_cache.pyro_2pl_map[e.example_id].disc,
                "diff_pyro_2pl": data_cache.pyro_2pl_map[e.example_id].diff,
            }
            for e in examples
        ]
    }
    await cache.set("/examples", response)
    return response


@api_app.get("/examples/plot_irt")
async def plot_examples_irt(cache: InMemoryCacheBackend = Depends(app_cache)):
    key = to_cache_key("/examples/plot_irt")
    in_cache = await cache.get(key)
    if in_cache is not None:
        log.info("Loaded from cache: /examples/plot_irt")
        return in_cache

    rows = []
    for qid in data_cache.squad_lookup.keys():
        rows.append(
            {
                "example_id": qid,
                "disc": data_cache.pyro_2pl_map[qid].disc,
                "diff": data_cache.pyro_2pl_map[qid].diff,
                "text": data_cache.squad_lookup[qid]["question"],
                "article": data_cache.qid_to_article[qid],
            }
        )
    df = pd.DataFrame(rows)
    brush = alt.selection(type="interval", empty="all")
    legend_selection = alt.selection_multi(fields=["article"], bind="legend", empty="all")
    base = alt.Chart(df)
    points = (
        base.mark_point().encode(
            x=alt.X("diff", title="Difficulty"),
            y=alt.Y("disc", title="Discriminability"),
            color=alt.Color(
                "article", title="Wikipedia Article", legend=alt.Legend(symbolLimit=40)
            ),
            tooltip=alt.Tooltip(["article", "text"]),
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.05)),
        )
    ).properties(width=800, height=500)
    top_hist = (
        base.mark_area()
        .encode(
            x=alt.X("diff", bin=alt.Bin(maxbins=50), stack=None, title=""),
            y=alt.Y("count()", stack=None, title=""),
            color=alt.Color("article"),
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.05)),
        )
        .properties(height=80, width=800)
    )
    right_hist = (
        base.mark_area().encode(
            x=alt.X("count()", stack=None, title=""),
            y=alt.Y("disc", bin=alt.Bin(maxbins=50), stack=None, title=""),
            color=alt.Color("article"),
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.05)),
        )
    ).properties(width=80, height=500)
    graphic = top_hist & (points | right_hist)

    ranked_text = (
        alt.Chart(df)
        .mark_text()
        .encode(y=alt.Y("row_number:O", axis=None))
        .transform_window(
            row_number="row_number()", sort=[alt.SortField("diff", order="descending")]
        )
        .transform_filter(brush)
        .transform_filter(legend_selection)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < 20)
    )
    article = ranked_text.encode(text="article").properties(width=50, title="Article")
    question = ranked_text.encode(text="text").properties(width=400, title="Text")
    diff = ranked_text.encode(text=alt.Text("diff", format=".3")).properties(
        width=50, title="Difficulty"
    )
    disc = ranked_text.encode(text=alt.Text("disc", format=".3")).properties(
        width=50, title="Discriminability"
    )
    table = article | question | diff | disc
    chart = (graphic & table).add_selection(brush, legend_selection)
    response = chart.to_dict()
    await cache.set(key, response)
    return response


@api_app.post("/examples/plot")
async def plot_examples(
    request: PlotSubmissionsRequest,
    db: SessionLocal = Depends(get_db),
    cache: InMemoryCacheBackend = Depends(app_cache),
):
    key = to_cache_key({"path": "/examples/plot", "payload": request.dict()})
    in_cache = await cache.get(key)
    if in_cache is not None:
        log.info("Loaded from cache: /examples/plot")
        return in_cache

    submissions = db.query(Submission).filter(Submission.submission_id.in_(request.submission_ids))
    rows = []
    for sub in submissions:
        scored_predictions = data_cache.predictions.scored_predictions[sub.submission_id][
            "exact_match"
        ]
        for qid in data_cache.squad_lookup.keys():
            score = scored_predictions[qid]
            rows.append(
                {
                    "example_id": qid,
                    "disc": data_cache.pyro_2pl_map[qid].disc,
                    "diff": data_cache.pyro_2pl_map[qid].diff,
                    "text": data_cache.squad_lookup[qid]["question"],
                    "submission_id": sub.submission_id,
                    "submission_name": sub.name,
                    "score": score,
                    "correct": "Correct" if score == 1 else "Wrong",
                    "article": data_cache.qid_to_article[qid],
                }
            )
    df = pd.DataFrame(rows)
    brush = alt.selection(type="interval")
    points = (
        (
            alt.Chart(df)
            .mark_point(opacity=0.2)
            .encode(
                x=alt.X("diff", title="Difficulty"),
                y=alt.Y("disc", title="Discriminability"),
                color=alt.Color(
                    "correct:N",
                    sort=["Correct", "Wrong"],
                    title="Correct",
                    scale=alt.Scale(domain=["Correct", "Wrong"]),
                ),
                facet=alt.Facet("submission_name", title="Model Name", columns=2),
                tooltip=alt.Tooltip(["article", "text"]),
            )
        )
        .add_selection(brush)
        .properties(width=300, height=300)
    )
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            color=alt.Color(
                "submission_name",
                title="Model Name",
                scale=alt.Scale(domain=list(df.submission_name.unique())),
            ),
            y=alt.Y("submission_name", title="Model Name"),
            x=alt.X("sum(score)", title="Number of Examples Correct"),
        )
        .transform_filter(brush)
    )
    ranked_text = (
        alt.Chart(df)
        .mark_text()
        .encode(y=alt.Y("row_number:O", axis=None))
        .transform_window(row_number="row_number()")
        .transform_filter(brush)
        .transform_window(rank="rank(row_number)")
        .transform_filter(alt.datum.rank < 20)
    )
    article = ranked_text.encode(text="article").properties(width=50, title="Article")
    question = ranked_text.encode(text="text").properties(width=400, title="Text")
    diff = ranked_text.encode(text=alt.Text("diff", format=".3")).properties(
        width=50, title="Difficulty"
    )
    disc = ranked_text.encode(text=alt.Text("disc", format=".3")).properties(
        width=50, title="Discriminability"
    )
    table = alt.hconcat(article, question, diff, disc)
    chart = (
        (points & bars & table)
        .resolve_legend(color="independent")
        .configure_legend(labelOpacity=1, symbolOpacity=1)
    )
    response = chart.to_dict()
    await cache.set(key, response)
    return response


@api_app.get("/metrics/plot")
async def plot_metrics(cache: InMemoryCacheBackend = Depends(app_cache)):
    in_cache = await cache.get("/metrics/plot")
    if in_cache is not None:
        log.info("Loaded from cache: /metrics/plot")
        return in_cache

    chart = (
        alt.Chart(data_cache.model_df)
        .mark_bar()
        .encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=50)), y=alt.Y("count()"), fill="metric:N",)
    )
    faceted_chart = (
        alt.ConcatChart(
            columns=2,
            concat=[
                chart.transform_filter(alt.datum.metric == value).properties(title=value)
                for value in data_cache.model_df.metric.unique()
            ],
        )
        .resolve_scale(x="independent", y="independent")
        .resolve_axis(x="independent", y="independent")
    )
    response = faceted_chart.to_dict()
    await cache.set("/metrics/plot", response)
    return response


@api_app.post("/ranks/plot")
async def plot_ranks(
    submissions_request: Optional[PlotSubmissionsRequest] = None,
    cache: InMemoryCacheBackend = Depends(app_cache),
):
    key = to_cache_key({"path": "/ranks/plot", "args": submissions_request.dict()})
    in_cache = await cache.get(key)
    if in_cache is not None:
        log.info("Loaded from cache: /ranks/plot")
        return in_cache
    else:
        if submissions_request is None:
            df = data_cache.model_df
        else:
            df = data_cache.model_df
            df = df[df.model_id.isin(submissions_request.submission_ids)]
        base = (
            alt.Chart(df)
            .mark_line()
            .encode(x=alt.X("metric:N"), y=alt.Y("rank:Q"), color="model_id:N", tooltip="name",)
            .transform_window(
                rank="rank()", sort=[alt.SortField("score", "descending")], groupby=["metric"],
            )
            .properties(width=200, height=200)
        )

        all_charts = []
        for metric_1, metric_2 in itertools.combinations(data_cache.model_df.metric.unique(), 2):
            all_charts.append(
                base.transform_filter(
                    (alt.expr.datum.metric == metric_1) | (alt.expr.datum.metric == metric_2)
                )
            )

        chart = alt.hconcat()
        for line in chunks(all_charts, 2):
            chart |= alt.vconcat(*line)
        response = chart.to_dict()
        await cache.set(key, response)
        return response


app = FastAPI()
app.mount("/api/1.0", api_app)
app.mount("/static-data", StaticFiles(directory=DATA_ROOT / "static-data"), name="static-data")


@app.on_event("startup")
async def startup_event():
    initialize_cache()
    mem_cache = InMemoryCacheBackend()
    caches.set(CACHE_KEY, mem_cache)


@app.on_event("shutdown")
async def on_shutdown():
    await close_caches()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
