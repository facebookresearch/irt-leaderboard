"""Copyright (c) Facebook, Inc. and its affiliates."""
import random
import subprocess
from pathlib import Path
from typing import Optional

import luigi
from functional import pseq
from luigi.local_target import LocalTarget
from pedroai.io import read_json, read_jsonlines, write_json, write_jsonlines
from sklearn.model_selection import train_test_split

from leaderboard import codalab, power
from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import create_leaderboard_splits, squad_pred_scores_to_jsonlines
from leaderboard.irt.evaluate import evaluate_irt_model
from leaderboard.irt.model_svi import SVISquadIrt
from leaderboard.linear import eval_vw, to_vw
from leaderboard.log import get_logger
from leaderboard.stats import Sampling, StatTest, run_parallel_test
from leaderboard.topics import TopicModel
from leaderboard.www.database import export_submissions

log = get_logger(__name__)

IRT_FAMILIES = ["pyro", "stan", "no_irt"]
IRT_TYPES = ["1PL", "2PL", "3PL", "no_irt_pl"]
FEATURE_SETS = list(conf["vw"].keys())


def generate_vw_dir_path(*, irt_family: str, irt_type: str, feature_set: Optional[str]):
    base_path = Path("data/linear") / irt_family / irt_type
    if feature_set is None:
        return base_path
    else:
        return base_path / feature_set


def shell(command: str):
    log.info("SHELL: %s", command)
    subprocess.run(command, shell=True, check=True)


class Database(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(conf["db"])


class LeaderboardJsonlines(luigi.Task):
    fold = luigi.Parameter()

    def requires(self):
        return Database()

    def run(self):
        if self.fold == "dev":
            submissions = export_submissions()
        elif self.fold == "test":
            submissions = squad_pred_scores_to_jsonlines("test")
        else:
            raise ValueError("Invalid fold")
        write_jsonlines(conf["squad"]["leaderboard"][self.fold], submissions)

    def output(self):
        return luigi.LocalTarget(conf["squad"]["leaderboard"][self.fold])


class SquadV2(luigi.ExternalTask):
    def output(self):
        return [
            luigi.LocalTarget(DATA_ROOT / conf["squad"]["train_v1"]),
            luigi.LocalTarget(DATA_ROOT / conf["squad"]["train_v2"]),
            luigi.LocalTarget(DATA_ROOT / conf["squad"]["dev_v2"]),
        ]


class SugawaraFeatures(luigi.ExternalTask):
    def output(self):
        return [
            luigi.LocalTarget(DATA_ROOT / "data" / "squad" / "datasets" / "squad-easy-subset.json"),
            luigi.LocalTarget(DATA_ROOT / "data" / "squad" / "datasets" / "squad-hard-subset.json"),
        ]


class SquadOutV2(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(DATA_ROOT / conf["squad"]["out_v2"])


class PairedSquadDevTest(luigi.Task):
    def requires(self):
        yield SquadOutV2()

    def run(self):
        codalab.pair_squad_dev_test()

    def output(self):
        return luigi.LocalTarget(DATA_ROOT / conf["squad"]["dev_to_test"])


class LeaderboardSubmissionPredictions(luigi.ExternalTask):
    fold = luigi.ChoiceParameter(choices=["dev", "test"])

    def output(self):
        return luigi.LocalTarget(DATA_ROOT / conf["squad"]["submission_predictions"][self.fold])


class SquadTopics(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(DATA_ROOT / "data" / "squad-topics.txt")


class LeaderboardTrainTestSplit(luigi.Task):
    train_size = luigi.FloatParameter(0.9, significant=False)
    seed = luigi.IntParameter(42, significant=False)
    fold = luigi.Parameter()

    def requires(self):
        yield LeaderboardJsonlines(fold=self.fold)

    def run(self):
        create_leaderboard_splits(self.fold, seed=self.seed, train_size=self.train_size)

    def output(self):
        return luigi.LocalTarget(conf["squad"]["leaderboard_splits"][self.fold])


IRT_EVALUATION_TYPES = ["full", "heldout", "subject_heldout"]


class SplitDevSubjects(luigi.Task):
    """
    Split dev subjects for the sampling experiment
    """

    seed = luigi.IntParameter(42, significant=False)

    def requires(self):
        yield LeaderboardJsonlines(fold="dev")

    def run(self):
        random.seed(self.seed)
        subjects = read_jsonlines(conf["squad"]["leaderboard"]["dev"])
        train, heldout = train_test_split(subjects, train_size=0.8)
        train_ids = [s["submission_id"] for s in train]
        heldout_ids = [s["submission_id"] for s in heldout]
        out = {"train": train_ids, "heldout": heldout_ids}
        write_json(DATA_ROOT / conf["stability"]["dev_subject_splits"], out)

    def output(self):
        return luigi.LocalTarget(DATA_ROOT / conf["stability"]["dev_subject_splits"])


class SamplingIRTModel(luigi.Task):
    device = luigi.Parameter(default="cpu", significant=False)

    def requires(self):
        yield SplitDevSubjects()

    def run(self):
        model = SVISquadIrt(
            data_path=conf["squad"]["leaderboard"]["dev"],
            model="2PL",
            evaluation="subject_heldout",
        )
        model.train(device=self.device)
        model.save(
            DATA_ROOT
            / conf["irt"]["squad"]["dev"]["pyro"]["2PL"]["subject_heldout"]
            / "parameters.json"
        )

    def output(self):
        return luigi.LocalTarget(
            DATA_ROOT
            / conf["irt"]["squad"]["dev"]["pyro"]["2PL"]["subject_heldout"]
            / "parameters.json"
        )


class PyroTrainIRT(luigi.Task):
    irt_type = luigi.ChoiceParameter(choices=IRT_TYPES)
    evaluation = luigi.ChoiceParameter(choices=IRT_EVALUATION_TYPES)
    device = luigi.Parameter(default="cpu", significant=False)
    fold = luigi.ChoiceParameter(choices=["dev", "test"])

    def requires(self):
        yield LeaderboardJsonlines(fold=self.fold)
        yield LeaderboardTrainTestSplit(fold=self.fold)

    def run(self):
        model = SVISquadIrt(
            data_path=conf["squad"]["leaderboard"][self.fold],
            model=self.irt_type,
            evaluation=self.evaluation,
        )
        model.train(device=self.device)
        model.save(
            DATA_ROOT
            / conf["irt"]["squad"][self.fold]["pyro"][self.irt_type][self.evaluation]
            / "parameters.json"
        )

    def output(self):
        return luigi.LocalTarget(
            DATA_ROOT
            / conf["irt"]["squad"][self.fold]["pyro"][self.irt_type][self.evaluation]
            / "parameters.json"
        )


class EvaluateIRT(luigi.Task):
    irt_type = luigi.ChoiceParameter(choices=IRT_TYPES)
    evaluation = luigi.ChoiceParameter(choices=IRT_EVALUATION_TYPES)
    device = luigi.Parameter(default="cpu", significant=False)
    fold = luigi.ChoiceParameter(choices=["dev", "test"])

    def requires(self):
        yield PyroTrainIRT(
            evaluation=self.evaluation, irt_type=self.irt_type, device=self.device, fold=self.fold,
        )

    def run(self):
        evaluate_irt_model(
            evaluation=self.evaluation,
            model_type=self.irt_type,
            model_family="pyro",
            fold=self.fold,
        )

    def output(self):
        base_dir = (
            DATA_ROOT / conf["irt"]["squad"][self.fold]["pyro"][self.irt_type][self.evaluation]
        )
        return [
            luigi.LocalTarget(base_dir / "report.json"),
            luigi.LocalTarget(base_dir / "roc.pdf"),
            luigi.LocalTarget(base_dir / "precision_recall.pdf"),
        ]


class AllIRTEvaluations(luigi.WrapperTask):
    device = luigi.Parameter(default="cpu", significant=False)

    def requires(self):
        folds = ["dev"]
        if Path(conf["squad"]["submission_predictions"]["test"]).exists():
            folds.append("test")

        for fold in folds:
            for irt_type in IRT_TYPES:
                for evaluation in IRT_EVALUATION_TYPES:
                    if "no_irt" in irt_type or "no_irt" in evaluation:
                        continue

                    # Only need a trained model for the CAT sampling stability
                    # experiment
                    if evaluation == "subject_heldout":
                        if irt_type == "2PL" and fold == "dev":
                            yield PyroTrainIRT(
                                evaluation=evaluation,
                                irt_type=irt_type,
                                device=self.device,
                                fold=fold,
                            )
                    else:
                        yield EvaluateIRT(
                            evaluation=evaluation, irt_type=irt_type, device=self.device, fold=fold,
                        )


class VWTrainTestData(luigi.Task):
    irt_family = luigi.ChoiceParameter(choices=IRT_FAMILIES)
    irt_type = luigi.ChoiceParameter(choices=IRT_TYPES)
    feature_set = luigi.ChoiceParameter(choices=FEATURE_SETS)

    def requires(self):
        if self.irt_family == "pyro":
            # Use the best pyro model
            yield PyroTrainIRT(irt_type=self.irt_type, evaluation="full")
        elif self.irt_family == "stan":
            raise NotImplementedError
        elif self.irt_family == "no_irt":
            pass
        else:
            raise ValueError(f"Invalid model family: {self.irt_family}")
        yield LeaderboardTrainTestSplit(fold="dev")
        # yield SquadTopics()
        yield SugawaraFeatures()
        yield SquadV2()
        yield LeaderboardSubmissionPredictions(fold="dev")

    def run(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        to_vw(
            irt_family=self.irt_family,
            irt_type=self.irt_type,
            feature_set=self.feature_set,
            out_dir=out_dir,
        )

    def output(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        return (
            LocalTarget(out_dir / "train.vw.txt"),
            LocalTarget(out_dir / "test.vw.txt"),
        )


class VWHyperTune(luigi.Task):
    irt_family = luigi.ChoiceParameter(choices=IRT_FAMILIES)
    irt_type = luigi.ChoiceParameter(choices=IRT_TYPES)
    feature_set = luigi.ChoiceParameter(choices=FEATURE_SETS)

    def requires(self):
        return VWTrainTestData(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )

    def run(self):
        # pylint: disable=line-too-long
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        args = [
            "python",
            "scripts/vw-hyperopt.py",
            "--max_evals",
            "20",
            "--train",
            str(out_dir / "train.vw.txt"),
            "--holdout",
            str(out_dir / "test.vw.txt"),
            "--best_vw_command_file",
            str(out_dir / "best-vw-command.json"),
            "--vw_space",
            "'--learning_rate=.001..10~L --passes=5 --l2=1e-8..1e-1~L0 -b=20,21,22,23 --link=logistic --loss_function=logistic'",
            "--out_dir",
            str(out_dir),
            "--plot",
        ]
        interactions = conf["vw"][self.feature_set].get("interactions")
        if interactions is not None:
            args.append("--additional_cmd")
            interaction_args = []
            for feature_interaction in interactions:
                interaction_args.append("--interactions")
                interaction_args.append(str(feature_interaction))
            additional_cmd = " ".join(interaction_args)
            args.append(f"'{additional_cmd}'")
        shell(" ".join(args))

    def output(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        return LocalTarget(out_dir / "best-vw-command.json")


class VWModel(luigi.Task):
    irt_family = luigi.ChoiceParameter(choices=IRT_FAMILIES)
    irt_type = luigi.ChoiceParameter(choices=IRT_TYPES)
    feature_set = luigi.ChoiceParameter(choices=FEATURE_SETS)

    def requires(self):
        return (
            VWTrainTestData(
                irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
            ),
            VWHyperTune(
                irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
            ),
        )

    def run(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        best_vw_command = read_json(out_dir / "best-vw-command.json")["best_vw_command"]
        shell(best_vw_command.replace("current.model", "best.model"))
        test_args = [
            "vw",
            "--testonly",
            "-d",
            str(out_dir / "test.vw.txt"),
            "-i",
            str(out_dir / "best.model"),
            "-p",
            str(out_dir / "test.pred.txt"),
            "-k",
            "--loss_function=logistic",
            "--link=logistic",
        ]
        shell(" ".join(test_args))

    def output(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        return (
            LocalTarget(out_dir / "best.model"),
            LocalTarget(out_dir / "test.pred.txt"),
        )


class EvaluateVWModel(luigi.Task):
    irt_family = luigi.ChoiceParameter(choices=IRT_FAMILIES)
    irt_type = luigi.ChoiceParameter(choices=IRT_TYPES)
    feature_set = luigi.ChoiceParameter(choices=FEATURE_SETS)

    def requires(self):
        return VWModel(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )

    def run(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        eval_vw(
            test_examples_file=out_dir / "test.vw.txt",
            test_pred_file=out_dir / "test.pred.txt",
            report_dir=out_dir,
            name=f"vw-{self.feature_set}",
        )

    def output(self):
        out_dir = generate_vw_dir_path(
            irt_family=self.irt_family, irt_type=self.irt_type, feature_set=self.feature_set,
        )
        return (
            LocalTarget(out_dir / "report.json"),
            LocalTarget(out_dir / "roc.pdf"),
            LocalTarget(out_dir / "precision_recall.pdf"),
        )


class AllVWModels(luigi.WrapperTask):
    def requires(self):
        for f in FEATURE_SETS:
            features = conf["vw"][f]["features"]
            if "irt" in features:
                for irt_type in ["1PL", "2PL", "3PL"]:
                    # Only care about pyro model for now
                    yield EvaluateVWModel(irt_family="pyro", irt_type=irt_type, feature_set=f)
            else:
                yield EvaluateVWModel(feature_set=f, irt_family="no_irt", irt_type="no_irt_pl")


class SquadInTopicFormat(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(conf["squad"]["mallet_v2"])


class RunTopicModel(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return SquadInTopicFormat()

    def run(self):
        output_dir = Path(conf["topic"][self.name]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        model = TopicModel(
            num_topics=conf["topic"][self.name]["num_topics"],
            input_file=conf["squad"]["mallet_v2"],
            output_dir=output_dir,
        )
        model.train()

    def output(self):
        output_dir = Path(conf["topic"][self.name]["output_dir"])
        return [
            luigi.LocalTarget(output_dir / "parameters.json"),
            luigi.LocalTarget(output_dir / "mallet.topic_keys"),
            luigi.LocalTarget(output_dir / "mallet.topic_distributions"),
            luigi.LocalTarget(output_dir / "mallet.model"),
            luigi.LocalTarget(output_dir / "input_data.mallet"),
            luigi.LocalTarget(output_dir / "mallet.state.gz"),
        ]


class AllTopicModels(luigi.WrapperTask):
    def requires(self):
        for name in conf["topic"].keys():
            yield RunTopicModel(name=name)


class StatisticalTests(luigi.Task):
    resources = {"parallel": 1}
    irt_type = luigi.ChoiceParameter(choices=["1PL", "2PL", "3PL"])
    sampling = luigi.EnumParameter(enum=Sampling)
    percent = luigi.IntParameter()
    fold = luigi.ChoiceParameter(choices=["dev", "test"])

    def requires(self):
        yield LeaderboardSubmissionPredictions(fold=self.fold)

    def run(self):
        pseq(list(StatTest)).map(
            lambda t: run_parallel_test(
                StatTest(t),
                irt_type=self.irt_type,
                sampling=self.sampling,
                percent=self.percent,
                fold=self.fold,
            )
        ).list()

    def output(self):
        output_dir = (
            Path("data/stats")
            / f"fold={self.fold}"
            / f"sampling={self.sampling.value}"  # pylint: disable=no-member
            / f"percent={self.percent}"
        )
        return [luigi.LocalTarget(output_dir / f"{test.value}.json") for test in list(StatTest)]


class AllStatisticalTests(luigi.WrapperTask):
    def requires(self):
        folds = ["dev"]
        if Path(conf["squad"]["submission_predictions"]["test"]).exists():
            folds.append("test")

        for fold in folds:
            if fold == "dev":
                for percent in [5, 10, 25, 50, 75, 100]:
                    for sampling in list(Sampling):
                        yield StatisticalTests(
                            irt_type="3PL", sampling=sampling, percent=percent, fold=fold,
                        )
            elif fold == "test":
                yield StatisticalTests(
                    irt_type="3PL", sampling=Sampling.RANDOM, percent=100, fold=fold
                )


class McNemarStatisticalPower(luigi.Task):
    resources = {"parallel": 1}

    def run(self):
        power.mcnemar("data/power/mcnemar_trials.json")

    def output(self):
        return luigi.LocalTarget("data/power/mcnemar_trials.json")
