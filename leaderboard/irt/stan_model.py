"""Copyright (c) Facebook, Inc. and its affiliates."""
from pathlib import Path

import pystan
import typer

from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import IrtModelType
from leaderboard.irt.model import SquadIrt
from leaderboard.log import get_logger

log = get_logger(__name__)

STAN_MODELS = {
    "1PL": Path("./leaderboard/irt/stan/1PL.stan"),
    "2PL": Path("./leaderboard/irt/stan/2PL.stan"),
    "3PL": Path("./leaderboard/irt/stan/3PL.stan"),
}

stan_app = typer.Typer()


class StanSquadIrt(SquadIrt):
    # pylint: disable=arguments-differ
    def __init__(
        self,
        *,
        data_path: Path,
        model: str,
        evaluation: str,
        metric: str = "exact_match",
        squad_fold: str = "dev",
    ):
        super().__init__(
            data_path=data_path,
            model=model,
            evaluation=evaluation,
            metric=metric,
            squad_fold=squad_fold,
        )
        self._stan_model = None
        self._stan_fit = None

    @property
    def indexing(self):
        return 1

    @property
    def model_type(self):
        if self.model == "1PL":
            irt_model_type = IrtModelType.stan_1pl
        elif self.model == "2PL":
            irt_model_type = IrtModelType.stan_2pl
        else:
            raise ValueError(f"Invalid model type: {self.model}")
        return irt_model_type.value

    def train(self, iterations=1000, chains=4):
        self._stan_model = pystan.StanModel(file=STAN_MODELS[self.model].open())
        self._stan_fit = self._stan_model.sampling(
            data=self.squad_data, iter=iterations, chains=chains
        )

    def export(self):
        results = {}
        results = {
            key: list(value.mean(0))
            for key, value in self._stan_fit.extract(["alpha", "beta"]).items()
        }

        try:
            results["disc"] = list(self._stan_fit.extract(["gamma"])["gamma"].mean(0))
        except ValueError:
            log.info("Missing parameter: gamma")

        try:
            for key, value in self._stan_fit.extract(
                ["mu_beta", "sigma_beta", "sigma_gamma"]
            ).items():
                value = list(value)
                if len(value) == 1:
                    value = value[0]
                results[key] = value
        except ValueError:
            log.info("Missing parameter: mu_beta, sigma_beta or sigma_gamma")
        results["example_ids"] = self.ix_to_example_id
        results["model_ids"] = self.ix_to_model_id
        results["irt_model"] = self.model_type
        return results


@stan_app.command()
def train(model: str, evaluation: str, data: Path, fold: str = "dev"):
    if model not in STAN_MODELS.keys():
        raise ValueError("Invalid model type")
    irt_squad_model = StanSquadIrt(
        data_path=data, model=model, evaluation=evaluation, squad_fold=fold
    )
    irt_squad_model.train()
    irt_squad_model.save(
        DATA_ROOT / conf["irt"]["squad"][fold]["stan"][model][evaluation] / "parameters.json"
    )
