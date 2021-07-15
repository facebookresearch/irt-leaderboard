"""Copyright (c) Facebook, Inc. and its affiliates."""
import os
from pathlib import Path

import toml

DATA_ROOT = Path(os.environ.get("LEADERBOARD_DATA_ROOT", "./"))

with open(DATA_ROOT / "config.toml") as f:
    conf = toml.load(f)
