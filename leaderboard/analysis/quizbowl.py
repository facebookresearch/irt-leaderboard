"""Copyright (c) Facebook, Inc. and its affiliates."""
from pedroai.io import read_json


def load_quizbowl():
    dataset = read_json("data/qanta.mapped.2018.04.18.json")["questions"]
    return [q for q in dataset if q["page"] is not None]
