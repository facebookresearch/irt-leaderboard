"""Copyright (c) Facebook, Inc. and its affiliates."""
import abc
import random
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyro
import typer
from pedroai.io import read_json, read_jsonlines, write_json

from leaderboard.config import conf
from leaderboard.data import IrtParsed, IrtResults, LeaderboardPredictions
from leaderboard.irt.model_svi import SVISquadIrt
from leaderboard.log import get_logger

log = get_logger(__name__)
rank_stability_app = typer.Typer()


PREDICTIONS: Dict[str, LeaderboardPredictions] = {}
SUBJECT_IDS = {}
ITEM_IDS = {}


def get_leaderboard_predictions(fold: str):
    if fold in PREDICTIONS:
        return PREDICTIONS[fold].scored_predictions
    else:
        PREDICTIONS[fold] = LeaderboardPredictions.parse_file(
            conf["squad"]["submission_predictions"][fold]
        )
        return PREDICTIONS[fold].scored_predictions


def get_subject_ids(fold: str):
    if fold in SUBJECT_IDS:
        return SUBJECT_IDS[fold]
    else:
        predictions = get_leaderboard_predictions(fold)
        subject_ids = list(predictions.keys())
        SUBJECT_IDS[fold] = subject_ids
        return subject_ids


def get_item_ids(fold: str):
    if fold in ITEM_IDS:
        return ITEM_IDS[fold]
    else:
        predictions = get_leaderboard_predictions(fold)
        subject_id = list(predictions.keys())[0]
        item_ids = list(predictions[subject_id]["exact_match"])
        ITEM_IDS[fold] = item_ids
        return item_ids


SUBMISSIONS = {}


def get_all_submissions(file: str):
    if file in SUBMISSIONS:
        return SUBMISSIONS[file]
    else:
        submissions_by_line = read_jsonlines(file)
        id_to_submission = {}
        for submission in submissions_by_line:
            subject_id = submission["submission_id"]
            id_to_submission[subject_id] = submission

        SUBMISSIONS[file] = id_to_submission
        return SUBMISSIONS[file]


class Ranker(ABC):
    def __init__(self, fold: str) -> None:
        super().__init__()
        self._fold = fold
        self._predictions = get_leaderboard_predictions(fold)

    @abc.abstractmethod
    def rank(self, subject_ids: List[str], item_ids: List[str]):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class ClassicalRanker(Ranker):
    def __init__(self, fold: str, metric: str = "exact_match") -> None:
        super().__init__(fold=fold)
        self._metric = metric

    def rank(self, subject_ids: List[str], item_ids: List[str]):
        subject_scores = defaultdict(int)
        for sid in subject_ids:
            subject_scored_preds = self._predictions[sid][self._metric]
            for item_id in item_ids:
                subject_scores[sid] += subject_scored_preds[item_id]
        sorted_subjects = sorted(subject_scores.items(), key=lambda s: s[1], reverse=True)
        return {s[0]: s[1] for s in sorted_subjects}

    def reset(self):
        pass


class IrtRanker(Ranker):
    def __init__(self, *, irt_type: str, fold: str) -> None:
        super().__init__(fold=fold)
        self._fold = fold
        self._irt_type = irt_type
        self._irt_model = None
        self._id_to_subject = get_all_submissions(conf["squad"]["leaderboard"][fold])

    def _create_input(self, subject_ids: List[str], item_ids: List[str]):
        subjects = []
        for sid in subject_ids:
            subject_data = self._id_to_subject[sid]
            items = {}
            for item_id in item_ids:
                items[item_id] = subject_data["predictions"][item_id]
            subjects.append(
                {"predictions": items, "submission_id": sid, "name": subject_data["name"],}
            )
        return subjects

    def rank(self, subject_ids: List[str], item_ids: List[str]):
        python_data = self._create_input(subject_ids, item_ids)
        self._irt_model = SVISquadIrt(
            data_path="/tmp/dummy",
            evaluation="full",
            model=self._irt_type,
            python_data=python_data,
        )
        self._irt_model.train()
        results = IrtParsed.from_irt_results(IrtResults(**self._irt_model.export()))
        subjects = []
        for sid in subject_ids:
            subjects.append({"subject_id": sid, "ability": results.model_stats[sid].skill})
        sorted_subjects = sorted(subjects, key=lambda s: s["ability"], reverse=True)
        return {s["subject_id"]: s["ability"] for s in sorted_subjects}

    def reset(self):
        pyro.clear_param_store()
        self._irt_model = None


def run_trial(*, fold: str, size: int, irt_ranker: IrtRanker, classical_ranker: ClassicalRanker):
    subject_ids = get_subject_ids(fold)
    item_ids = get_item_ids(fold)

    sampled_items = random.sample(item_ids, 2 * size)
    a_sample_items = sampled_items[:size]
    classical_ranker.reset()
    irt_ranker.reset()
    a_classical_ranks = classical_ranker.rank(subject_ids, a_sample_items)
    a_irt_ranks = irt_ranker.rank(subject_ids, a_sample_items)

    b_sample_items = sampled_items[size:]
    classical_ranker.reset()
    irt_ranker.reset()
    b_classical_ranks = classical_ranker.rank(subject_ids, b_sample_items)
    b_irt_ranks = irt_ranker.rank(subject_ids, b_sample_items)

    rows = []
    for sid in subject_ids:
        rows.append(
            {
                "subject_id": sid,
                "a_classical": a_classical_ranks[sid],
                "a_irt": a_irt_ranks[sid],
                "b_classical": b_classical_ranks[sid],
                "b_irt": b_irt_ranks[sid],
                "size": size,
                "fold": fold,
            }
        )
    return pd.DataFrame(rows)


CACHED = {}


def run_configured_simulation(
    *, trial_size: int, trial_id: int, irt_type: str, fold: str, output_dir: str, metric: str,
):
    file = Path(output_dir) / f"rank_stability_{trial_size}_{trial_id}.feather"

    if file.exists():
        log.info("Simualtion exists, skipping for size=%s trial=%s", trial_size, trial_id)

    log.info("Running size=%s trial=%s", trial_size, trial_id)
    if "irt" not in CACHED:
        CACHED["irt"] = IrtRanker(irt_type=irt_type, fold=fold)
    irt_ranker = CACHED["irt"]

    if "classical" not in CACHED:
        CACHED["classical"] = ClassicalRanker(fold=fold, metric=metric)
    classical_ranker = CACHED["classical"]
    results = run_trial(
        size=trial_size, irt_ranker=irt_ranker, classical_ranker=classical_ranker, fold=fold,
    )
    results["trial_id"] = trial_id
    results.to_feather(file)


@rank_stability_app.command()
def configure(
    *,
    fold: str,
    irt_type: str,
    output_dir: str,
    step_size: int = 100,
    n_trials: int = 10,
    # Just hard set this, I know squad size
    max_size: int = 5000,
):
    output_dir = Path(output_dir)
    size = step_size
    commands = []
    while size < max_size:
        for t in range(n_trials):
            commands.append(f"sbatch bin/slurm-run-rank-stability.sh {size} {t} {output_dir}")
        size += step_size

    output_dir.mkdir(exist_ok=True)
    write_json(output_dir / "config.json", {"fold": fold, "irt_type": irt_type})
    contents = "\n".join(commands)
    with open(output_dir / "run.sh", "w") as f:
        f.write(contents)


@rank_stability_app.command()
def run(
    *, trial_size: int, trial_id: int, output_dir: str,
):
    log.info("Preparing simulation")

    output_dir = Path(output_dir)
    config = read_json(output_dir / "config.json")
    fold = config["fold"]
    irt_type = config["irt_type"]

    log.info("Starting job")
    run_configured_simulation(
        trial_size=trial_size,
        trial_id=trial_id,
        irt_type=irt_type,
        fold=fold,
        output_dir=output_dir,
        metric="exact_match",
    )
