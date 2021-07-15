"""Copyright (c) Facebook, Inc. and its affiliates."""
import abc
import enum
from csv import DictReader
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pedroai.io import read_json, read_jsonlines, safe_file, write_json

from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import IrtResults, LeaderboardSplits
from leaderboard.log import get_logger

log = get_logger(__name__)


def load_subset_squad_data(data: Path):
    qid_to_ix, ix_to_qid = {}, []
    student_to_obs, question_to_obs, observations = [], [], []

    with data.open() as fp:
        reader = DictReader(fp)

        for i, row in enumerate(reader):
            qid_to_ix[row["qid"]] = i
            ix_to_qid.append(row["qid"])

            observations.extend(
                [int(row["albert"]), int(row["bert"]), int(row["bidaf"]), int(row["xlnet"]),]
            )
            student_to_obs.extend([1, 2, 3, 4])
            question_to_obs.extend([i + 1, i + 1, i + 1, i + 1])

    squad_data = {
        "J": 4,
        "K": len(ix_to_qid),
        "N": len(observations),
        "jj": student_to_obs,
        "kk": question_to_obs,
        "y": observations,
    }

    return qid_to_ix, ix_to_qid, squad_data


class EvaluationType(enum.Enum):
    FULL = "full"
    HELDOUT = "heldout"
    SUBJECT_HELDOUT = "subject_heldout"


class SquadIrt(abc.ABC):
    def __init__(
        self,
        *,
        model: str,
        data_path: Path,
        evaluation: str,
        metric="exact_match",
        squad_fold: str = "dev",
        # Optionally, pass in the already read in data
        python_data: Optional[List] = None,
    ):
        super().__init__()
        self.evaluation = EvaluationType(evaluation)
        self.model = model
        self.data_path = data_path
        self.student_to_obs, self.question_to_obs, self.observations = [], [], []
        self.squad_fold = squad_fold
        if python_data is None:
            self.all_submissions = read_jsonlines(data_path)
        else:
            log.info("Using passed in python data")
            self.all_submissions = python_data
        self.splits = LeaderboardSplits.parse_file(
            DATA_ROOT / conf["squad"]["leaderboard_splits"][squad_fold]
        )
        self.test_items = {(item.model_id, item.example_id) for item in self.splits.test}
        # Create a canonical order
        self.example_ids = set(self.all_submissions[0]["predictions"].keys())
        self.example_id_to_ix, self.ix_to_example_id = {}, {}
        self.model_id_to_ix = {}
        self.ix_to_model_id = {}
        for i, idx in enumerate(self.example_ids, start=self.indexing):
            self.example_id_to_ix[idx] = i
            self.ix_to_example_id[i] = idx

        if self.evaluation == EvaluationType.SUBJECT_HELDOUT:
            subject_splits = read_json(DATA_ROOT / conf["stability"]["dev_subject_splits"])
            self.train_subject_ids = set(subject_splits["train"])
            n_subjects = len(subject_splits["train"]) + len(subject_splits["heldout"])

            log.info("Loading split subjects")
            log.info(f"Total: {n_subjects} N_TRAIN: {len(self.train_subject_ids)}")
        else:
            self.train_subject_ids = None

        self.n_submissions = 0
        i = self.indexing
        for submission in self.all_submissions:
            model_id = submission["submission_id"]
            if self.evaluation == EvaluationType.SUBJECT_HELDOUT:
                if model_id in self.train_subject_ids:
                    self.n_submissions += 1
                    self.model_id_to_ix[model_id] = i
                    self.ix_to_model_id[i] = model_id
                else:
                    continue
            else:
                self.n_submissions += 1
                self.model_id_to_ix[model_id] = i
                self.ix_to_model_id[i] = model_id
            for pred in submission["predictions"].values():
                example_id = pred["example_id"]
                if self.evaluation == EvaluationType.FULL:
                    self.observations.append(pred["scores"][metric])
                    self.student_to_obs.append(i)
                    self.question_to_obs.append(self.example_id_to_ix[example_id])
                elif self.evaluation == EvaluationType.HELDOUT:
                    if (model_id, example_id) not in self.test_items:
                        self.observations.append(pred["scores"][metric])
                        self.student_to_obs.append(i)
                        self.question_to_obs.append(self.example_id_to_ix[example_id])
                elif self.evaluation == EvaluationType.SUBJECT_HELDOUT:
                    self.observations.append(pred["scores"][metric])
                    self.student_to_obs.append(i)
                    self.question_to_obs.append(self.example_id_to_ix[example_id])
                else:
                    raise ValueError("Invalid evaluation type")
            i += 1

        if len({len(self.model_id_to_ix), len(self.ix_to_model_id), self.n_submissions}) != 1:
            raise ValueError("Number of submissions does not match, there must be duplicates")
        self.squad_data = {
            "J": self.n_submissions,
            "K": len(self.ix_to_example_id),
            "N": len(self.observations),
            "jj": self.student_to_obs,
            "kk": self.question_to_obs,
            "y": self.observations,
        }

    @property
    @abc.abstractmethod
    def indexing(self):
        pass

    @abc.abstractmethod
    def export(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def save(self, output_path: Union[str, Path]):
        # validate the results dictionary
        write_json(safe_file(output_path), IrtResults(**self.export()).dict())
