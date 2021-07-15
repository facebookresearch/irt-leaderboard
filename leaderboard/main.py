"""Copyright (c) Facebook, Inc. and its affiliates."""
import typer

from leaderboard import plots
from leaderboard.codalab import codalab_app
from leaderboard.comparison_stability import comparison_stability_app
from leaderboard.data import data_app
from leaderboard.irt.model_svi import pyro_app
from leaderboard.irt.stan_model import stan_app
from leaderboard.irt_stats import irt_train_app
from leaderboard.linear import features_app
from leaderboard.power import power_app
from leaderboard.rank_stability import rank_stability_app
from leaderboard.squad import squad_app
from leaderboard.stats import stats_app
from leaderboard.topics import topic_app

app = typer.Typer()
app.add_typer(squad_app, name="squad")
app.add_typer(codalab_app, name="cl")
app.add_typer(comparison_stability_app, name="comparison_stability")
app.add_typer(rank_stability_app, name="rank_stability")
app.add_typer(stats_app, name="stats")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(power_app, name="power")
app.add_typer(pyro_app, name="pyro")
app.add_typer(stan_app, name="stan")
app.add_typer(topic_app, name="topic")
app.add_typer(irt_train_app, name="irt_train")
app.command(name="plot")(plots.main)


if __name__ == "__main__":
    app()
