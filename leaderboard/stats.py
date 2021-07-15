"""
Copyright (c) Facebook, Inc. and its affiliates.
Run experiments for running different statistical significance tests

- IRT Significance
- Paired student t-test
- Sign test
- McNemar's test
- Wilcoxon signed rank

The tests are run with:
- All the data
- Random subsets of the data
- Selected subsets (eg, by IRT parameters)
"""
import enum
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm
import typer
from functional import pseq
from mlxtend import evaluate
from pedroai.io import safe_file, write_json
from pydantic import BaseModel
from rich.console import Console
from scipy import stats
from scipy.stats import norm

from leaderboard.config import conf
from leaderboard.data import IrtParsed, LeaderboardPredictions

stats_app = typer.Typer()
console = Console()

TESTS = ["sem", "mcnemar", "wilcoxon", "student_t", "see"]


class StatTest(enum.Enum):
    SEM = "sem"
    MCNEMAR = "mcnemar"
    WILCOXON = "wilcoxon"
    STUDENT_T = "student_t"
    SEE = "see"


class PairedStats(BaseModel):
    model_a: str
    model_b: str
    key: str
    score_a: float
    score_b: float
    statistic: Optional[float]
    pvalue: Optional[float]
    test: str
    max_score: float
    min_score: float
    diff: float
    fold: str
    metric: str
    metadata: Optional[Dict]


def mcnemar_test(model_a_array: np.ndarray, model_b_array: np.ndarray):
    """
    McNemar's test operates on contingency tables, which we need bo build first.
    """
    both_correct = 0
    both_wrong = 0
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    for a, b in zip(model_a_array, model_b_array):
        if a == 1.0 and b == 1.0:
            both_correct += 1
        elif a == 0.0 and b == 0.0:
            both_wrong += 1
        elif a == 1.0 and b == 0.0:
            a_correct_b_wrong += 1
        elif a == 0.0 and b == 1.0:
            a_wrong_b_correct += 1
        else:
            raise ValueError(f"Invalid predictions: {a}, {b}")
    contingency_table = np.array(
        [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    )
    return evaluate.mcnemar(ary=contingency_table, corrected=True)


def kuder_richardson_formula_20(student_item_metrix: np.ndarray):
    # pylint: disable=line-too-long
    """
    Compute the Kuder Richardson Formula 20

    The measure checks the internal consistency of measurements
    over dichotomous choices. It is a measure of reliability which
    is computed as part of the Standard Error of Measurement in
    classical statistical testing theory.

    student_item_matrix should be a 2d array so that entry
    x_ij = score of student i on item j

    Reference:
    https://www.real-statistics.com/reliability/internal-consistency-reliability/kuder-richardson-formula-20/
    """
    if student_item_metrix.ndim != 2:
        raise ValueError(f"student_item_matrix must be 2d, but was: {student_item_metrix.ndim}")
    n_students, n_items = student_item_metrix.shape
    if n_students == 0 or n_items == 0:
        raise ValueError("Need at least one student and one item")
    scores = student_item_metrix.sum(axis=1)
    correct_by_item = student_item_metrix.sum(axis=0)
    # p
    p_correct_by_item = correct_by_item / n_students
    # q
    p_wrong_by_item = 1 - p_correct_by_item
    # pq
    pq = p_correct_by_item * p_wrong_by_item
    score_variance = np.var(scores)
    reliability = (n_items / (n_items - 1)) * (1 - pq.sum() / score_variance)
    return reliability, scores


def calculate_info(skill, disc, diff):
    temp = disc * (skill - diff)
    return (disc * np.exp(-temp) / ((1 + np.exp(-temp)) ** 2)).sum()


def calculate_see(skill, disc, diff):
    temp = disc * (skill - diff)
    info_j = disc * np.exp(-temp) / ((1 + np.exp(-temp)) ** 2)
    return 1 / np.sqrt(info_j.sum())


class Sampling(enum.Enum):
    RANDOM = "random"
    IRT_DIFF = "irt_diff"
    IRT_DISC = "irt_disc"
    ACCURACY = "accuracy"


class EvaluationData:
    def __init__(
        self,
        *,
        irt_type: str,
        sampling: Sampling,
        fold: str,
        percent: float = None,
        parallel: bool = False,
        metric: str = "exact_match",
    ) -> None:
        self._fold = fold
        self._irt_type = irt_type
        self._sampling = sampling
        self._percent = percent
        self._fraction = percent / 100
        self._metric = metric
        self.parallel = parallel
        self.data = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"][fold])
        self.parsed_irt = IrtParsed.from_irt_file(
            Path(conf["irt"]["squad"][fold]["pyro"][irt_type]["full"]) / "parameters.json"
        )
        self._sampled_example_ids = self._sample_examples(self.parsed_irt.example_ids)

    def _sample_examples(self, example_ids: List[str]):
        console.log(
            "Initializing sample cache, sampling=", self._sampling, "percent=", self._percent,
        )
        if self._sampling == Sampling.RANDOM:
            return self._random_sampling(example_ids)
        elif self._sampling == Sampling.IRT_DIFF:
            return self._diff_sampling(example_ids)
        elif self._sampling == Sampling.IRT_DISC:
            return self._disc_sampling(example_ids)
        elif self._sampling == Sampling.ACCURACY:
            return self._accuracy_sampling(example_ids)
        else:
            raise ValueError(f"Invalid sampling type: {self._sampling}")

    def _random_sampling(self, example_ids: List[str]):
        if self._percent == 100:
            return example_ids
        else:
            n = len(example_ids)
            console.log("N=", n)
            sample_size = int(n * self._fraction)
            return random.sample(self.parsed_irt.example_ids, sample_size)

    def _diff_sampling(self, example_ids: List[str]):
        diffs = []
        for ex_id in example_ids:
            example_stats = self.parsed_irt.example_stats[ex_id]
            diffs.append((ex_id, example_stats.diff))
        n = len(example_ids)
        sample_size = int(n * self._fraction)
        # We want the most difficult examples
        diffs = sorted(diffs, reverse=True, key=lambda t: t[1])
        return [t[0] for t in diffs[:sample_size]]

    def _disc_sampling(self, example_ids: List[str]):
        discs = []
        for ex_id in example_ids:
            example_stats = self.parsed_irt.example_stats[ex_id]
            discs.append((ex_id, example_stats.disc))
        n = len(example_ids)
        sample_size = int(n * self._fraction)
        # We want the most discrminating examples
        discs = sorted(discs, reverse=True, key=lambda t: t[1])
        return [t[0] for t in discs[:sample_size]]

    def _accuracy_sampling(self, example_ids: List[str]):
        example_accuracy = defaultdict(float)
        for model_scores in self.data.scored_predictions.values():
            for ex_id in example_ids:
                example_accuracy[ex_id] += model_scores[self._metric][ex_id]
        n_models = len(self.data.scored_predictions)
        for ex_id, correct in example_accuracy.items():
            example_accuracy[ex_id] = correct / n_models

        items = [(ex_id, example_accuracy[ex_id]) for ex_id in example_ids]
        # We want the lowest accuracy (hardest) examples
        sorted_items = sorted(items, key=lambda t: t[1])
        n = len(example_ids)
        sample_size = int(n * self._fraction)
        return [t[0] for t in sorted_items[:sample_size]]

    def create_standard_error_of_estimation(self):
        discs = []
        diffs = []
        for example_id in self._sampled_example_ids:
            example_stats = self.parsed_irt.example_stats[example_id]
            diffs.append(example_stats.diff)
            discs.append(example_stats.disc)
        discs = np.array(discs)
        diffs = np.array(diffs)

        irt_model_data = {}
        for mid in self.parsed_irt.model_ids:
            model_stats = self.parsed_irt.model_stats[mid]
            see = calculate_see(model_stats.skill, discs, diffs)
            info = calculate_info(model_stats.skill, discs, diffs)
            irt_model_data[mid] = {
                "skill": model_stats.skill,
                "error": see,
                "info": info,
                "mid": mid,
            }

        def error_of_estimation(model_a: str, model_b: str):
            model_a_data = irt_model_data[model_a]
            model_a_error = model_a_data["error"]
            model_a_skill = model_a_data["skill"]

            model_b_data = irt_model_data[model_b]
            model_b_error = model_a_data["error"]
            model_b_skill = model_b_data["skill"]

            diff_error = np.sqrt(model_a_error ** 2 + model_b_error ** 2)
            skill_diff = abs(model_a_skill - model_b_skill)
            diff_dist = norm(loc=0, scale=diff_error)
            # The normal distribution is symmetric and we want a two tailed
            # test, so this is fine.
            prob = 2 * diff_dist.cdf(-skill_diff)
            return model_a_skill, model_b_skill, skill_diff, prob

        return error_of_estimation

    def extract_paired_data(self, model_a: str, model_b: str,) -> Tuple[List[float], List[float]]:
        model_a_preds = self.data.scored_predictions[model_a]
        model_b_preds = self.data.scored_predictions[model_b]

        example_a_ids = set(model_a_preds[self._metric].keys())
        example_b_ids = set(model_b_preds[self._metric].keys())
        if len(example_a_ids.symmetric_difference(example_b_ids)) != 0:
            raise ValueError("Mismatched examples in predictions")
        model_a_array = []
        model_b_array = []
        for exid in self._sampled_example_ids:
            model_a_array.append(model_a_preds[self._metric][exid])
            model_b_array.append(model_b_preds[self._metric][exid])
        return model_a_array, model_b_array

    def extract_student_item_matrix(self):
        model_ids = list(self.data.scored_predictions.keys())
        matrix = []
        for mid in model_ids:
            model_preds = self.data.scored_predictions[mid][self._metric]
            row = []
            for exid in self._sampled_example_ids:
                row.append(model_preds[exid])
            matrix.append(row)
        matrix = np.array(matrix)
        return matrix

    def create_standard_error_of_measure(self):
        student_item_matrix = self.extract_student_item_matrix()
        reliability, scores = kuder_richardson_formula_20(student_item_matrix)
        stdev = np.std(scores)
        sem = stdev * np.sqrt(1 - reliability)

        def standard_error_of_measure(model_a_array: np.ndarray, model_b_array: np.ndarray):
            model_a_score = sum(model_a_array)
            model_b_score = sum(model_b_array)
            diff = abs(model_a_score - model_b_score)
            dist = norm(loc=0, scale=sem)
            return sem, 2 * dist.sf(diff)

        return standard_error_of_measure

    def permutation_test(self):
        # TODO: add permutation test
        # evaluate.permutation_test
        pass

    def compute_model_scores(self, model_id: str):
        if self._percent == 100:
            # If using full dataset, then can just fetch computed score
            return self.data.model_scores[model_id][self._metric]
        else:
            score = 0
            example_scores = self.data.scored_predictions[model_id][self._metric]
            for example_id in self._sampled_example_ids:
                score += example_scores[example_id]
            return score / len(self._sampled_example_ids)

    def run_test(self, test_type: StatTest):
        """
        The Wilcoxon test is a non-parametric test for comparing matched samples.
        It is a non-parameteric alternative to the Student T-test, which assumes
        that the difference is normally distributed.

        The test determines if two dependent samples are from the same distribution
        by comparing population mean rank differences.
        """
        if test_type == StatTest.WILCOXON:
            stat_test = stats.wilcoxon
        elif test_type == StatTest.STUDENT_T:
            stat_test = stats.ttest_rel
        elif test_type == StatTest.MCNEMAR:
            stat_test = mcnemar_test
        elif test_type == StatTest.SEM:
            stat_test = self.create_standard_error_of_measure()
        elif test_type == StatTest.SEE:
            stat_test = self.create_standard_error_of_estimation()
        else:
            raise ValueError(f"Invalid test: {test_type}")

        results = []
        completed = set()
        model_ids = list(self.data.scored_predictions.keys())
        model_pairs = list(itertools.product(model_ids, model_ids))
        if self.parallel:
            tqdm_position = TESTS.index(test_type.value)
        else:
            tqdm_position = None
        for model_a, model_b in tqdm.tqdm(
            model_pairs, position=tqdm_position, desc=f"Test: {test_type.value}"
        ):
            if model_a != model_b:
                key = tuple(sorted([model_a, model_b]))
                if key in completed:
                    continue
                completed.add(key)

                model_a_array, model_b_array = self.extract_paired_data(model_a, model_b)
                model_a_array = np.array(model_a_array)
                model_b_array = np.array(model_b_array)

                model_a_score = self.compute_model_scores(model_a)
                model_b_score = self.compute_model_scores(model_b)
                if (model_a_array - model_b_array).sum() == 0:
                    results.append(
                        PairedStats(
                            model_a=model_a,
                            model_b=model_b,
                            key=" ".join(sorted([model_a, model_b])),
                            score_a=model_a_score,
                            score_b=model_b_score,
                            max_score=max(model_a_score, model_b_score),
                            min_score=min(model_a_score, model_b_score),
                            diff=abs(model_a_score - model_b_score),
                            statistic=None,
                            pvalue=None,
                            test=test_type.value,
                            metric=self._metric,
                            fold="dev",
                        )
                    )
                    continue

                if test_type == StatTest.SEE:
                    model_a_skill, model_b_skill, statistic, pvalue = stat_test(model_a, model_b)
                    metadata = {
                        "model_a_skill": model_a_skill,
                        "model_b_skill": model_b_skill,
                    }
                else:
                    statistic, pvalue = stat_test(model_a_array, model_b_array)
                    metadata = None
                results.append(
                    PairedStats(
                        model_a=model_a,
                        model_b=model_b,
                        key=" ".join(sorted([model_a, model_b])),
                        score_a=model_a_score,
                        score_b=model_b_score,
                        max_score=max(model_a_score, model_b_score),
                        min_score=min(model_a_score, model_b_score),
                        diff=abs(model_a_score - model_b_score),
                        statistic=statistic,
                        pvalue=pvalue,
                        test=test_type.value,
                        metric=self._metric,
                        fold="dev",
                        metadata=metadata,
                    )
                )
        return results


def run_parallel_test(
    test: StatTest, *, irt_type: str, sampling: Sampling, percent: int, fold: str
):
    data = EvaluationData(
        irt_type=irt_type, sampling=sampling, percent=percent, parallel=True, fold=fold
    )
    results = [r.dict() for r in data.run_test(test)]
    write_json(
        safe_file(
            f"data/stats/fold={fold}/sampling={sampling.value}/percent={percent}/{test.value}.json"
        ),
        {
            "results": results,
            "test": test.value,
            "sampling": sampling.value,
            "percent": percent,
            "irt_type": irt_type,
        },
    )
    return True


@stats_app.command()
def compute(irt_type: str, sampling: Sampling, percent: int, fold: str):
    pseq(list(StatTest)).map(
        lambda t: run_parallel_test(
            StatTest(t), irt_type=irt_type, sampling=sampling, percent=percent, fold=fold
        )
    ).list()
