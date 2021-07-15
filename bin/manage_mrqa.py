# Copyright (c) Facebook, Inc. and its affiliates.
import subprocess
from pathlib import Path

import toml
import typer
from pedroai.io import read_json, write_jsonlines
from rich.console import Console
from rich.progress import track

app = typer.Typer()
console = Console()


def download_bundle(bundle_id: str, output_path: str):
    command = f"cl download -o {output_path} {bundle_id}"
    console.log("Command:", command)
    subprocess.run(command, shell=True, check=True)


def score_bundle(dataset_file: str, bundle_file: str, output_file: str):
    command = f"python bin/mrqa_official_eval.py {dataset_file} {bundle_file} {output_file}"
    console.log("Command:", command)
    subprocess.run(command, shell=True, check=True)


@app.command()
def download():
    with open("config/mrqa.toml") as f:
        conf = toml.load(f)
    output_dir = Path(conf["mrqa"]["pred_dir"])
    output_dir.mkdir(exist_ok=True)
    for bundle_id in track(conf["mrqa"]["bundle_ids"]):
        path = str(output_dir / f"{bundle_id}.json")
        download_bundle(bundle_id, path)


@app.command()
def score():
    with open("config/mrqa.toml") as f:
        conf = toml.load(f)
    pred_dir = Path(conf["mrqa"]["pred_dir"])
    data_dir = Path(conf["mrqa"]["data_dir"])
    score_dir = Path(conf["mrqa"]["score_dir"])

    for bundle_id in track(conf["mrqa"]["bundle_ids"]):
        for dataset in conf["mrqa"]["datasets"]:
            score_bundle(
                data_dir / f"{dataset}.jsonl",
                pred_dir / f"{bundle_id}.json",
                score_dir / f"{dataset}_{bundle_id}.json",
            )


@app.command()
def export():
    with open("config/mrqa.toml") as f:
        conf = toml.load(f)
    score_dir = Path(conf["mrqa"]["score_dir"])
    submissions = []
    for bundle_id in track(conf["mrqa"]["bundle_ids"]):
        task_scores = {}
        scored_predictions = {}
        for dataset in conf["mrqa"]["datasets"]:
            dataset_scores = read_json(score_dir / f"{dataset}_{bundle_id}.json")
            task_scores[dataset] = dataset_scores["exact_match"]
            for qid, qid_score in dataset_scores["scores_exact_match"].items():
                item_id = f"{dataset}_{qid}"
                scored_predictions[item_id] = {
                    "scores": {"exact_match": qid_score, "f1": 0},
                    "submission_id": bundle_id,
                    "example_id": item_id,
                }
        submissions.append(
            {
                "predictions": scored_predictions,
                "submission_id": bundle_id,
                "name": bundle_id,
                "task_scores": task_scores,
            }
        )
    write_jsonlines(conf["mrqa"]["leaderboard"], submissions)


if __name__ == "__main__":
    app()
