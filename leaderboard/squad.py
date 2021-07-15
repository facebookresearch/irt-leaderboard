"""Copyright (c) Facebook, Inc. and its affiliates."""
from typing import List, Optional

import typer
from rich.progress import track
from rich.console import Console
from pedroai.io import read_jsonlines, write_jsonlines

from leaderboard.analysis.squad import SquadPredictionData
from leaderboard.data import create_leaderboard_splits
from leaderboard.www import database

squad_app = typer.Typer()
console = Console()


@squad_app.command("plot")
def squad_plot(models: List[str]):
    data = SquadPredictionData(models)
    data.plot_comparison()


@squad_app.command("anova")
def squad_anova(models: List[str]):
    data = SquadPredictionData(models)
    data.repeated_measures_anova()


@squad_app.command("permutation")
def squad_permutation(models: List[str]):
    data = SquadPredictionData(models)
    data.permutation_test()


@squad_app.command("dist")
def squad_dist(models: List[str]):
    data = SquadPredictionData(models)
    data.dist_info()


@squad_app.command("csv")
def squad_to_csv(out_path: str, models: List[str]):
    data = SquadPredictionData(models)
    data.to_csv(out_path)


@squad_app.command()
def init_db(limit_submissions: Optional[int] = None, skip_tests: bool = False):
    # TODO: Move this out of the squad command
    database.build_db(limit_submissions=limit_submissions, skip_tests=skip_tests)


@squad_app.command()
def export_to_irt(out_path: str):
    submissions = database.export_submissions()
    write_jsonlines(out_path, submissions)


@squad_app.command()
def item_splits(fold: str, seed: int = 42, train_size: float = 0.9):
    create_leaderboard_splits(fold, seed=seed, train_size=train_size)


@squad_app.command()
def leaderboard_to_pyirt(data_path: str, output_path: str, metric: str = 'exact_match'):
    """Convert the SQuAD leaderboard.jsonlines data to py-irt format"""
    output = []
    for line in track(read_jsonlines(data_path)):
        subject_id = line['submission_id']
        # Extra field, just passing meta info
        name = line['name']
        responses = {}
        for item_id, scores in line['predictions'].items():
            responses[item_id] = scores['scores'][metric]
        output.append({
            'subject_id': subject_id,
            'responses': responses,
            'name': name
        })
    console.log(f'Writing output to: {output_path}')
    write_jsonlines(output_path, output)
