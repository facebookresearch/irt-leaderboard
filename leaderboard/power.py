"""Copyright (c) Facebook, Inc. and its affiliates."""
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from mlxtend import evaluate
from pedroai.io import write_json
from rich.console import Console

console = Console()
power_app = typer.Typer()


def simulate(p_correct: float, n_trials: int, n_points: int, delta: float, alpha: float = 0.05):
    all_agreements = []
    power = []
    antipower = []
    for _ in range(n_trials):
        baseline_p_correct = p_correct
        other_p_correct = baseline_p_correct + delta
        contingency_table = np.zeros((2, 2))
        n_baseline_correct = 0
        n_other_correct = 0
        # Get predictions
        for _ in range(n_points):
            if np.random.random() < baseline_p_correct:
                baseline_table = 0
                n_baseline_correct += 1
            else:
                baseline_table = 1

            if np.random.random() < other_p_correct:
                other_table = 0
                n_other_correct += 1
            else:
                other_table = 1

            contingency_table[baseline_table][other_table] += 1

        agreement = contingency_table[0][0] + contingency_table[1][1]
        all_agreements.append(agreement / n_points)
        _, p = evaluate.mcnemar(ary=contingency_table)
        baseline_accuracy = n_baseline_correct / n_points
        other_accuracy = n_other_correct / n_points
        observed_effect = other_accuracy - baseline_accuracy
        if observed_effect > 0 and p <= alpha:
            power.append(1)
        else:
            power.append(0)

        if observed_effect <= 0 and p >= alpha:
            antipower.append(1)
        else:
            antipower.append(0)

    avg_power = np.mean(power)
    avg_antipower = np.mean(antipower)
    return {
        "delta": delta,
        "power": avg_power,
        "antipower": avg_antipower,
        "agreement": np.mean(all_agreements),
        "p_correct": p_correct,
    }


@power_app.command()
def mcnemar(out_dir: str, n_trials: int = 1000, dataset_size: int = 10000):
    trials = []
    # check every 5 points
    for p_correct in np.linspace(0.05, 0.95, 19):
        for delta in np.linspace(0, 0.05, 10):
            trials.append((p_correct, n_trials, dataset_size, delta))
    console.log("Running trials n=", len(trials))
    rows = Parallel(n_jobs=-1, verbose=11)(delayed(simulate)(*t) for t in trials)
    out_dir = Path(out_dir)
    df = pd.DataFrame(rows)
    out = {}
    out["df"] = df.to_dict()
    out["n_trials"] = n_trials
    out["dataset_size"] = dataset_size
    write_json(out_dir / "mcnemar_trials.json", out)
