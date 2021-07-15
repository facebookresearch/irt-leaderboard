"""Copyright (c) Facebook, Inc. and its affiliates."""
import os

from leaderboard.irt.model_svi import SVISquadIrt

# We didn't use the stan model so no use in testing it
# from leaderboard.irt.stan_model import StanSquadIrt


# def test_stan_irt_1pl_save():
#     irt = StanSquadIrt(
#         data_path="test_fixtures/leaderboard.jsonlines", model="1PL", evaluation="full"
#     )
#     irt.train(iterations=2, chains=1)
#     irt.save("/tmp/1PL_stan.json")
#     assert os.path.exists("/tmp/1PL_stan.json")


# def test_stan_irt_2pl_save():
#     irt = StanSquadIrt(
#         data_path="test_fixtures/leaderboard.jsonlines", model="2PL", evaluation="full"
#     )
#     irt.train(iterations=2, chains=1)
#     irt.save("/tmp/2PL_stan.json")
#     assert os.path.exists("/tmp/2PL_stan.json")


def test_pyro_irt_1pl_save():
    irt = SVISquadIrt(
        data_path="test_fixtures/leaderboard.jsonlines", model="1PL", evaluation="full"
    )
    irt.train(iterations=2)
    irt.save("/tmp/1PL_pyro.json")
    assert os.path.exists("/tmp/1PL_pyro.json")


def test_pyro_irt_2pl_save():
    irt = SVISquadIrt(
        data_path="test_fixtures/leaderboard.jsonlines", model="2PL", evaluation="full"
    )
    irt.train(iterations=2)
    irt.save("/tmp/2PL_pyro.json")
    assert os.path.exists("/tmp/2PL_pyro.json")


def test_pyro_irt_3pl_save():
    irt = SVISquadIrt(
        data_path="test_fixtures/leaderboard.jsonlines", model="3PL", evaluation="full"
    )
    irt.train(iterations=2)
    irt.save("/tmp/3PL_pyro.json")
    assert os.path.exists("/tmp/3PL_pyro.json")
