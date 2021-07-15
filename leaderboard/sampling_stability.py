"""Copyright (c) Facebook, Inc. and its affiliates."""
import abc
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import typer
from functional import pseq
from pedroai.io import read_json
from rich.progress import track
from scipy import optimize

from leaderboard.config import conf
from leaderboard.data import (
    IrtParsed,
    LeaderboardPredictions,
    load_squad_id_to_question,
)
from leaderboard.rank_stability import ClassicalRanker

sampling_app = typer.Typer()


class Sampler(abc.ABC):
    def __init__(self, fold: str):
        self._fold = fold
        self._squad = load_squad_id_to_question()
        self._item_ids = list(self._squad.keys())

    @abc.abstractmethod
    def sample(self, n_items: int):
        pass


class RandomSampler(Sampler):
    def sample(self, n_items: int):
        return random.sample(self._item_ids, n_items)


def create_sort_key(param_key: str):
    def sort_key(item_stats):
        if param_key == "high_disc":
            return item_stats.disc
        elif param_key == "low_disc":
            return -item_stats.disc
        elif param_key == "zero_disc":
            return -abs(item_stats.disc)
        elif param_key == "high_diff":
            return item_stats.diff
        elif param_key == "low_diff":
            return -item_stats.diff
        elif param_key == "zero_diff":
            return -abs(item_stats.diff)
        elif param_key == "high_disc_diff":
            return item_stats.disc + item_stats.diff
        else:
            raise ValueError("Invalid sort key")

    return sort_key


class IrtSampler(Sampler):
    def __init__(self, *, fold: str, param_key: str, irt_model: str):
        super().__init__(fold)
        self._param_key = param_key
        self._irt_model = IrtParsed.from_irt_file(
            Path(conf["irt"]["squad"][fold]["pyro"][irt_model]["full"]) / "parameters.json"
        )

    def sample(self, n_items: int):
        item_stats = {}
        for item_id in self._item_ids:
            item_stats[item_id] = self._irt_model.example_stats[item_id]

        sorted_items = sorted(
            item_stats.values(), key=create_sort_key(self._param_key), reverse=True
        )
        return [item.example_id for item in sorted_items[:n_items]]


def prob_2pl(*, skill: float, disc: float, diff: float) -> float:
    exponent = -disc * (skill - diff)
    return 1 / (1 + np.exp(exponent))


CACHE_PREDICTIONS = {}


def cached_leaderboard_predictions(fold: str) -> LeaderboardPredictions:
    if fold in CACHE_PREDICTIONS:
        return CACHE_PREDICTIONS[fold]
    else:
        preds = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"][fold])
        CACHE_PREDICTIONS[fold] = preds
        return preds


CACHE_IRT = {}


def cached_irt(irt_type: str) -> IrtParsed:
    if irt_type in CACHE_IRT:
        return CACHE_IRT[irt_type]
    else:
        irt_model = IrtParsed.from_irt_file(
            Path(conf["irt"]["squad"]["dev"]["pyro"][irt_type]["full"]) / "parameters.json"
        )
        CACHE_IRT[irt_type] = irt_model
        return irt_model


class InfoSampler(Sampler):
    def __init__(
        self, *, fold: str, irt_type: str, subject_ids: List[str],
    ):
        super().__init__(fold)
        if irt_type != "2PL":
            raise ValueError("Not impl for more than 2PL")
        self._chosen_items = set()
        self._irt_type = irt_type
        self._remaining_items = set(self._item_ids)
        self._subject_ids = subject_ids

    def estimate_theta(self, subject_id: str) -> float:
        responses = cached_leaderboard_predictions("dev").scored_predictions[subject_id][
            "exact_match"
        ]

        def likelihood(theta):
            total = 0
            n = 0
            correct = 0
            irt_model = cached_irt(self._irt_type)
            for item_id in self._chosen_items:
                item_stats = irt_model.example_stats[item_id]
                prob = prob_2pl(skill=theta, diff=item_stats.diff, disc=item_stats.disc)
                if responses[item_id] == 1:
                    if prob > 0.5:
                        correct += 1
                    total += np.log(prob)
                else:
                    if prob < 0.5:
                        correct += 1
                    total += np.log(1 - prob)
                n += 1
            return -total

        return optimize.minimize(likelihood, 0, method="L-BFGS-B").x[0]

    def compute_item_information(self, subject_skill: float, item_id: str) -> float:
        item_stats = cached_irt(self._irt_type).example_stats[item_id]
        prob = prob_2pl(skill=subject_skill, disc=item_stats.disc, diff=item_stats.diff)
        return item_stats.disc ** 2 * prob * (1 - prob)

    def compute_sum_information(self, subject_skills: Dict[str, float]) -> Dict[str, float]:
        item_infos = defaultdict(float)
        for skill in subject_skills.values():
            for item_id in self._remaining_items:
                item_infos[item_id] += self.compute_item_information(skill, item_id)
        return item_infos

    def initial_items(self, n_items: int):
        item_stats = {}
        for item_id in self._item_ids:
            item_stats[item_id] = cached_irt(self._irt_type).example_stats[item_id]

        sorted_items = sorted(item_stats.values(), key=lambda item: item.disc, reverse=True)
        return [item.example_id for item in sorted_items[:n_items]]

    def sample(self, n_items: int):
        if n_items < len(self._chosen_items):
            raise ValueError("cannot choose fewer items than last queried")
        elif n_items == len(self._chosen_items):
            return list(self._chosen_items)
        elif len(self._chosen_items) == 0:
            selected_items = self.initial_items(n_items)
            self._remaining_items = self._remaining_items - set(selected_items)
            self._chosen_items = self._chosen_items | set(selected_items)
            return selected_items
        else:
            subject_skills = {}
            thetas = pseq(self._subject_ids).map(self.estimate_theta).list()
            for subject_id, theta in zip(self._subject_ids, thetas):
                subject_skills[subject_id] = theta
                # subject_skills[subject_id] = self.estimate_theta(subject_id)
            item_information = list(self.compute_sum_information(subject_skills).items())
            sorted_items = sorted(item_information, key=lambda k: k[1], reverse=True)
            item_keys = [i[0] for i in sorted_items]
            needed_items = n_items - len(self._chosen_items)
            selected_items = item_keys[:needed_items]
            self._remaining_items = self._remaining_items - set(selected_items)
            self._chosen_items = self._chosen_items | set(selected_items)
            if len(self._chosen_items) != n_items:
                raise ValueError("mismatch")
            return list(self._chosen_items)


class Simulation:
    def __init__(self, max_size: int = 6000, step_size: int = 25, n_trials: int = 10):
        self._step_size = step_size
        self._n_trials = n_trials
        squad = load_squad_id_to_question()

        # dev_preds = LeaderboardPredictions.parse_file(
        #    conf["squad"]["submission_predictions"]["dev"]
        # )
        test_preds = LeaderboardPredictions.parse_file(
            conf["squad"]["submission_predictions"]["test"]
        )
        test_item_ids = set()
        for scored_preds in test_preds.scored_predictions.values():
            for item_id in scored_preds["exact_match"].keys():
                test_item_ids.add(item_id)
            break

        test_item_ids = list(test_item_ids)
        # squad_scores = load_squad_submissions(dev_preds)
        self._random_sampler = RandomSampler("dev")
        self._high_disc_sampler = IrtSampler(fold="dev", param_key="high_disc", irt_model="3PL")
        self._low_disc_sampler = IrtSampler(fold="dev", param_key="low_disc", irt_model="3PL")
        self._zero_disc_sampler = IrtSampler(fold="dev", param_key="zero_disc", irt_model="3PL")

        self._high_diff_sampler = IrtSampler(fold="dev", param_key="high_diff", irt_model="3PL")
        self._low_diff_sampler = IrtSampler(fold="dev", param_key="low_diff", irt_model="3PL")
        self._zero_diff_sampler = IrtSampler(fold="dev", param_key="zero_diff", irt_model="3PL")

        self._high_disc_diff_sampler = IrtSampler(
            fold="dev", param_key="high_disc_diff", irt_model="3PL"
        )

        self._dev_classic_ranker = ClassicalRanker("dev")
        self._test_classic_ranker = ClassicalRanker("test")
        mapping = read_json(conf["squad"]["dev_to_test"])
        # dev_to_test = mapping["dev_to_test"]
        test_to_dev = mapping["test_to_dev"]
        unfiltered_test_subjects = list(test_preds.scored_predictions.keys())

        self._dev_subjects = []
        self._test_subjects = []
        for test_id in unfiltered_test_subjects:
            if test_id in test_to_dev:
                self._dev_subjects.append(test_to_dev[test_id])
                self._test_subjects.append(test_id)
        self._n_dev_questions = min(max_size, len(squad))
        self._test_ranking = self._test_classic_ranker.rank(self._test_subjects, test_item_ids)
        self._remapped_test_ranking = {
            test_to_dev[test_id]: score for test_id, score in self._test_ranking.items()
        }
        self._info_sampler = InfoSampler(fold="dev", irt_type="2PL", subject_ids=self._dev_subjects)

    def run(self):
        random_df = self.create_random_df()
        irt_df = self.create_irt_df()
        info_df = self.create_info_df()
        output_dir = Path(conf["stability"]["sampling"])
        random_df.to_json(output_dir / "random_df.json")
        irt_df.to_json(output_dir / "irt_df.json")
        info_df.to_json(output_dir / "info_df.json")

        return {
            "info_df": info_df,
            "irt_df": irt_df,
            "random_df": random_df,
        }

    def create_random_df(self):
        rows = []
        for trial_size in track(range(self._step_size, self._n_dev_questions, self._step_size)):
            for trial_id in range(10):
                random_items = self._random_sampler.sample(trial_size)
                random_dev_ranking = self._dev_classic_ranker.rank(self._dev_subjects, random_items)
                rank_rows = []
                for sid in self._dev_subjects:
                    rank_rows.append(
                        {
                            "dev_id": sid,
                            "dev_random_score": random_dev_ranking[sid],
                            "test_score": self._remapped_test_ranking[sid],
                        }
                    )
                rank_df = pd.DataFrame(rank_rows)
                corr = rank_df.corr(method="kendall")
                rows.append(
                    {
                        "trial_size": trial_size,
                        "trial_id": trial_id,
                        "dev_random_to_test": corr.loc["dev_random_score"].test_score,
                    }
                )
        random_df = pd.DataFrame(rows).melt(id_vars=["trial_size", "trial_id"])
        return random_df

    def create_irt_df(self):
        rows = []
        for trial_size in track(range(self._step_size, self._n_dev_questions, self._step_size)):
            high_disc_items = self._high_disc_sampler.sample(trial_size)
            low_disc_items = self._low_disc_sampler.sample(trial_size)
            zero_disc_items = self._zero_disc_sampler.sample(trial_size)

            high_diff_items = self._high_diff_sampler.sample(trial_size)
            low_diff_items = self._low_diff_sampler.sample(trial_size)
            zero_diff_items = self._zero_diff_sampler.sample(trial_size)

            high_disc_diff_items = self._high_disc_diff_sampler.sample(trial_size)

            high_disc_dev_ranking = self._dev_classic_ranker.rank(
                self._dev_subjects, high_disc_items
            )
            low_disc_dev_ranking = self._dev_classic_ranker.rank(self._dev_subjects, low_disc_items)
            zero_disc_dev_ranking = self._dev_classic_ranker.rank(
                self._dev_subjects, zero_disc_items
            )

            high_diff_dev_ranking = self._dev_classic_ranker.rank(
                self._dev_subjects, high_diff_items
            )
            low_diff_dev_ranking = self._dev_classic_ranker.rank(self._dev_subjects, low_diff_items)
            zero_diff_dev_ranking = self._dev_classic_ranker.rank(
                self._dev_subjects, zero_diff_items
            )

            high_disc_diff_dev_ranking = self._dev_classic_ranker.rank(
                self._dev_subjects, high_disc_diff_items
            )

            rank_rows = []
            for sid in self._dev_subjects:
                rank_rows.append(
                    {
                        "dev_id": sid,
                        "dev_high_disc_score": high_disc_dev_ranking[sid],
                        "dev_low_disc_score": low_disc_dev_ranking[sid],
                        "dev_zero_disc_score": zero_disc_dev_ranking[sid],
                        "dev_high_diff_score": high_diff_dev_ranking[sid],
                        "dev_low_diff_score": low_diff_dev_ranking[sid],
                        "dev_zero_diff_score": zero_diff_dev_ranking[sid],
                        "dev_high_disc_diff_score": high_disc_diff_dev_ranking[sid],
                        "test_score": self._remapped_test_ranking[sid],
                    }
                )
            rank_df = pd.DataFrame(rank_rows)
            corr = rank_df.corr(method="kendall")
            rows.append(
                {
                    "trial_size": trial_size,
                    "trial_id": 0,
                    "dev_high_disc_to_test": corr.loc["dev_high_disc_score"].test_score,
                    #'dev_low_disc_to_test': corr.loc['dev_low_disc_score'].test_score,
                    #'dev_zero_disc_to_test': corr.loc['dev_zero_disc_score'].test_score,
                    "dev_high_diff_to_test": corr.loc["dev_high_diff_score"].test_score,
                    #'dev_low_diff_to_test': corr.loc['dev_low_diff_score'].test_score,
                    #'dev_zero_diff_to_test': corr.loc['dev_zero_diff_score'].test_score,
                    "dev_high_disc_diff_to_test": corr.loc["dev_high_disc_diff_score"].test_score,
                }
            )
        irt_df = pd.DataFrame(rows).melt(id_vars=["trial_size", "trial_id"])
        return irt_df

    def create_info_df(self):
        rows = []
        for trial_size in track(range(self._step_size, self._n_dev_questions, self._step_size)):
            info_items = self._info_sampler.sample(trial_size)
            info_dev_ranking = self._dev_classic_ranker.rank(self._dev_subjects, info_items)

            rank_rows = []
            for sid in self._dev_subjects:
                rank_rows.append(
                    {
                        "dev_id": sid,
                        "dev_info_score": info_dev_ranking[sid],
                        "test_score": self._remapped_test_ranking[sid],
                    }
                )
            rank_df = pd.DataFrame(rank_rows)
            corr = rank_df.corr(method="kendall")
            rows.append(
                {
                    "trial_size": trial_size,
                    "trial_id": 0,
                    "dev_info_to_test": corr.loc["dev_info_score"].test_score,
                }
            )
        info_df = pd.DataFrame(rows).melt(id_vars=["trial_size", "trial_id"])
        return info_df


@sampling_app.command()
def run(max_size: int = 6000, step_size: int = 25, n_trials: int = 10):
    simulation = Simulation(max_size=max_size, step_size=step_size, n_trials=n_trials)
    simulation.run()
