"""Copyright (c) Facebook, Inc. and its affiliates."""
import argparse
import datetime
import json
from typing import Dict, List, Tuple

import typer
from codalab.lib.bundle_cli import BundleCLI
from codalab.lib.codalab_manager import CodaLabManager
from functional import pseq
from pedroai.io import read_json, write_json
from tqdm import tqdm

from leaderboard.analysis.squad import load_squad
from leaderboard.config import conf
from leaderboard.data import LeaderboardPredictions, LeaderboardSubmissions, Submission
from leaderboard.log import get_logger
from leaderboard.squad_eval_v2 import get_raw_scores

codalab_app = typer.Typer()
log = get_logger(__name__)


def pair_squad_dev_test():
    out = read_json(conf["squad"]["out_v2"])
    dev_to_test = {}
    test_to_dev = {}
    for row in out["leaderboard"]:
        if row["submission"]["public"] and row["bundle"]["state"] == "ready":
            submit_info = json.loads(row["bundle"]["metadata"]["description"])
            submit_id = submit_info["submit_id"]
            test_id = submit_info["predict_id"]
            if test_id in test_to_dev or submit_id in dev_to_test:
                raise ValueError()
            else:
                test_to_dev[test_id] = submit_id
                dev_to_test[submit_id] = test_id
    write_json(
        conf["squad"]["dev_to_test"], {"dev_to_test": dev_to_test, "test_to_dev": test_to_dev},
    )


def extract_models_bundles() -> Tuple[List[Submission], List[Dict]]:
    entries = read_json(conf["squad"]["out_v2"])

    bundles = []
    skipped = []
    for row in entries:
        public = row["submission"]["public"]
        state = row["bundle"]["state"]
        if public and state == "ready":
            name = row["submission"]["description"]
            submit_info = json.loads(row["bundle"]["metadata"]["description"])
            submit_id = submit_info["submit_id"]
            bundles.append(
                Submission(
                    name=name,
                    submit_id=submit_id,
                    state=state,
                    public=public,
                    bundle_id=row["bundle"]["id"],
                    scores=row["scores"],
                    created=datetime.datetime.fromtimestamp(row["submission"]["created"]),
                    submitter=row["submission"]["user_name"],
                )
            )
        else:
            skipped.append(row)
    return bundles, skipped


SQUAD_DATASET = []


def compute_pred_scores(submit_id):
    if len(SQUAD_DATASET) == 0:
        SQUAD_DATASET.append(load_squad()[0])
    pred_file = f"data/squad/submissions/{submit_id}.json"
    predictions = read_json(pred_file)
    exact_scores, f1_scores = get_raw_scores(SQUAD_DATASET[0], predictions)
    return (submit_id, {"exact_match": exact_scores, "f1": f1_scores})


@codalab_app.command()
def download(skip_download: bool = False):
    submissions, skipped = extract_models_bundles()
    codalab_cli = BundleCLI(CodaLabManager())
    for s in tqdm(submissions):
        if not skip_download:
            codalab_cli.do_download_command(
                argparse.Namespace(
                    output_path=f"data/squad/submissions/{s.submit_id}.json",
                    target_spec=s.submit_id,
                    worksheet_spec=None,
                    force=True,
                )
            )

    with open("data/squad/submission_metadata.json", "w") as f:
        f.write(LeaderboardSubmissions(submissions=submissions).json())
    write_json("data/squad/skipped_submissions.json", {"submissions": skipped})


@codalab_app.command()
def score_submissions():
    leaderboard = LeaderboardSubmissions(**read_json(conf["squad"]["submission_metadata"]))
    submit_ids = [s.submit_id for s in leaderboard.submissions]
    log.info("Scoring predictions in parallel")
    scored_predictions = pseq(submit_ids).map(compute_pred_scores).dict()
    scores_by_submission = {}
    for submit_id, metrics in scored_predictions.items():
        submission_scores = {}
        for metric_name, scores_by_example in metrics.items():
            submission_scores[metric_name] = sum(scores_by_example.values()) / len(
                scores_by_example
            )
        scores_by_submission[submit_id] = submission_scores
    scores = LeaderboardPredictions(
        scored_predictions=scored_predictions, model_scores=scores_by_submission
    )
    write_json(conf["squad"]["submission_predictions"]["dev"], scores.dict())


@codalab_app.command()
def evalute(data_file: str, pred_file: str):
    with open(data_file) as f:
        dataset = json.load(f)["data"]

    with open(pred_file) as f:
        preds = json.load(f)

    exact_scores, f1_scores = get_raw_scores(dataset, preds)
    log.info("%s, %s", exact_scores, f1_scores)
