"""Copyright (c) Facebook, Inc. and its affiliates."""
from pathlib import Path
from typing import List, Optional

import torch
import typer

from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import IrtModelType
from leaderboard.irt.evaluate import evaluate_irt_model
from leaderboard.irt.model import SquadIrt
from leaderboard.irt.pyirt.multidim_one_param_logistic import MultidimOneParamLog
from leaderboard.irt.pyirt.one_param_logistic import OneParamLogWithExport
from leaderboard.irt.pyirt.three_param_logistic import ThreeParamLog
from leaderboard.irt.pyirt.two_param_logistic import TwoParamLog

# from leaderboard.irt.pyirt.multidim import MultidimensionalIRT

PYRO_MODELS = {
    "1PL": OneParamLogWithExport,
    "2PL": TwoParamLog,
    "3PL": ThreeParamLog,
    "MD1PL": MultidimOneParamLog,
}


pyro_app = typer.Typer()


class SVISquadIrt(SquadIrt):
    # pylint: disable=arguments-differ
    def __init__(
        self,
        data_path: Path,
        model: str,
        evaluation: str,
        metric="exact_match",
        squad_fold: str = "dev",
        python_data: Optional[List] = None,
        dims: int = 2,
    ):
        super().__init__(
            model=model,
            data_path=data_path,
            evaluation=evaluation,
            metric=metric,
            squad_fold=squad_fold,
            python_data=python_data,
        )
        self._pyro_model = None
        self._dims = dims

    @property
    def indexing(self):
        return 0

    @property
    def model_type(self):
        if self.model == "1PL":
            model_type = IrtModelType.pyro_1pl.value
        elif self.model == "2PL":
            model_type = IrtModelType.pyro_2pl.value
        elif self.model == "3PL":
            model_type = IrtModelType.pyro_3pl.value
        elif self.model == "MD1PL":
            model_type = IrtModelType.pyro_md1pl.value
        else:
            raise ValueError(f"Invalid model type: {self.model}")
        return model_type

    def train(self, iterations: int = 1000, device: str = "cpu") -> None:
        # pylint: disable=not-callable
        device = torch.device(device)
        self._pyro_model = PYRO_MODELS[self.model](
            priors="hierarchical",
            device=device,
            num_items=len(self.ix_to_example_id),
            num_models=self.n_submissions,
            dims=self._dims,
        )

        self._pyro_model.fit(
            torch.tensor(self.student_to_obs, dtype=torch.long, device=device),
            torch.tensor(self.question_to_obs, dtype=torch.long, device=device),
            torch.tensor(self.observations, dtype=torch.float, device=device),
            iterations,
        )

    def export(self):
        results = self._pyro_model.export()
        results["irt_model"] = self.model_type
        results["example_ids"] = self.ix_to_example_id
        results["model_ids"] = self.ix_to_model_id
        return results


@pyro_app.command()
def train(model: str, evaluation: str, data: Path, fold: str = "dev", device: str = "cpu"):
    if model not in PYRO_MODELS.keys():
        raise ValueError("Invalid model type")
    irt_squad_model = SVISquadIrt(data_path=data, model=model, evaluation=evaluation)
    irt_squad_model.train(device=device)
    irt_squad_model.save(
        DATA_ROOT / conf["irt"]["squad"][fold]["pyro"][model][evaluation] / "parameters.json"
    )


@pyro_app.command()
def evaluate(model: str, evaluation: str, squad_fold: str = "dev"):
    evaluate_irt_model(
        fold=squad_fold, evaluation=evaluation, model_family="pyro", model_type=model
    )
