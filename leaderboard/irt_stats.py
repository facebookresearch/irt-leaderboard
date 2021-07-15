"""Copyright (c) Facebook, Inc. and its affiliates."""
import time
from pathlib import Path
from typing import Optional

import typer
from pedroai.io import safe_file, write_json
from rich.console import Console

from leaderboard.config import conf
from leaderboard.irt.evaluate import (
    evaluate_irt_model_in_dir,
    evaluate_multidim_model_in_dir,
)
from leaderboard.irt.model_svi import SVISquadIrt

irt_train_app = typer.Typer()
console = Console()


@irt_train_app.command()
def train(
    irt_type: str,
    fold: str,
    output_dir: str,
    evaluation: str = "heldout",
    epochs: int = 1000,
    data_path: Optional[str] = None,
    device: str = "cpu",
):
    console.log("Fold:", fold, "IRT:", irt_type, "Data Path:", "Output:", output_dir)
    console.log("Preparing model")
    start_time = time.time()
    if data_path is None:
        data_path = conf["squad"]["leaderboard"][fold]
    irt_model = SVISquadIrt(data_path=data_path, evaluation=evaluation, model=irt_type)
    output_dir = Path(output_dir)
    console.log("Training model")
    irt_model.train(iterations=epochs, device=device)
    irt_model.save(output_dir / "parameters.json")
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Train time:", elapsed_time)
    console.log("Evaluating model")
    evaluate_irt_model_in_dir(
        evaluation=evaluation,
        model_type=irt_type,
        model_family="pyro",
        fold=fold,
        irt_base_dir=output_dir,
    )


@irt_train_app.command()
def multidim_train(
    fold: str,
    output_dir: str,
    evaluation: str = "full",
    dims: int = 2,
    epochs: int = 1000,
    device: str = "cpu",
    data_path: Optional[str] = None,
):
    irt_type = "MD1PL"
    console.log("Fold:", fold, "IRT:", irt_type, "Data Path:", data_path, "Output:", output_dir)
    console.log("Preparing model")
    start_time = time.time()
    if data_path is None:
        data_path = conf["squad"]["leaderboard"][fold]
    irt_model = SVISquadIrt(data_path=data_path, evaluation=evaluation, model=irt_type, dims=dims,)
    output_dir = Path(output_dir)
    console.log("Training model")
    irt_model.train(iterations=epochs, device=device)
    params = irt_model.export()
    output_path = output_dir / "parameters.json"
    write_json(safe_file(output_path), params)

    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Train time:", elapsed_time)
    console.log("Evaluating model")
    evaluate_multidim_model_in_dir(
        evaluation=evaluation,
        model_type=irt_type,
        model_family="pyro",
        fold=fold,
        irt_base_dir=output_dir,
    )
