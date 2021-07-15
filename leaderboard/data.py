"""
Copyright (c) Facebook, Inc. and its affiliates.
Types, containers, and parsers for all our data.
"""
import datetime
import enum
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, TypeVar

import typer
from pedroai.io import read_json, read_jsonlines, write_json
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from leaderboard.config import conf

Json = Dict[str, Any]


data_app = typer.Typer()


def load_quizbowl(*, limit: Optional[int] = None, train: bool = False, sentences: bool = False):
    with open("data/qanta.mapped.2018.04.18.json") as f:
        questions = json.load(f)["questions"]
        questions = [q for q in questions if q["page"] is not None]
        for q in questions[:limit]:
            if train:
                if "train" in q["fold"]:
                    if sentences:
                        for sent in extract_sentences(q):
                            yield sent
                    else:
                        yield q
            else:
                if sentences:
                    for sent in extract_sentences(q):
                        yield sent
                else:
                    yield q


def extract_sentences(question: Json) -> List[Json]:
    sentences = []
    qanta_id = question["qanta_id"]
    sentence_tokenizations = question["tokenizations"]
    for idx, (start, end) in enumerate(sentence_tokenizations):
        text = question["text"][start:end]
        sentences.append(
            {"text": text, "start": start, "end": end, "qanta_id": qanta_id, "sentence_idx": idx,}
        )
    return sentences


class Submission(BaseModel):
    name: str
    bundle_id: str
    submit_id: str
    public: bool
    state: str
    submitter: str
    scores: Dict[str, float]
    created: datetime.datetime


class LeaderboardSubmissions(BaseModel):
    submissions: List[Submission]


PredictionScores = Dict[str, float]


class LeaderboardPredictions(BaseModel):
    # NOTE: All scores are for the same fold, so if this is from dev data
    # both scores will be from dev data
    scored_predictions: Dict[str, Dict[str, PredictionScores]]
    model_scores: Dict[str, Dict[str, float]]


class SquadV2Answer(BaseModel):
    text: str
    answer_start: int


class SquadV2Question(BaseModel):
    question: str
    id: str
    answers: List[SquadV2Answer]
    is_impossible: bool


class SquadV2Paragraph(BaseModel):
    context: str
    qas: List[SquadV2Question]


class SquadV2Page(BaseModel):
    title: str
    paragraphs: List[SquadV2Paragraph]


class SquadV2(BaseModel):
    version: str
    data: List[SquadV2Page]


def load_squad_v2(file: str) -> SquadV2:
    return SquadV2.parse_obj(read_json(file))


class IrtModelType(enum.Enum):
    pyro_1pl = "pyro_1pl"
    stan_1pl = "stan_1pl"
    pyro_2pl = "pyro_2pl"
    stan_2pl = "stan_2pl"
    pyro_3pl = "pyro_3pl"
    multidim_2pl = "multidim_2pl"
    pyro_md1pl = "pyro_md1pl"


class IrtResults(BaseModel):
    irt_model: str
    ability: Optional[List[float]]
    disc: Optional[List[float]]
    diff: Optional[List[float]]
    lambdas: Optional[List[float]]
    example_ids: Optional[Dict[str, str]]
    model_ids: Optional[Dict[str, str]]


Vector = List[float]


class MultidimIrtResults(BaseModel):
    irt_model: str
    ability: Optional[List[Vector]]
    disc: Optional[List[Vector]]
    diff: Optional[List[Vector]]
    lambdas: Optional[List[Vector]]
    example_ids: Optional[Dict[str, str]]
    model_ids: Optional[Dict[str, str]]


T = TypeVar("T", bound="IrtParsed")


class ExampleStats(BaseModel):
    irt_model: str
    example_id: str
    diff: float
    disc: Optional[float]
    lambda_: Optional[float]


class MultidimExampleStats(BaseModel):
    irt_model: str
    example_id: str
    diff: Vector
    disc: Optional[Vector]
    lambda_: Optional[Vector]


class ModelStats(BaseModel):
    irt_model: str
    model_id: str
    skill: float


class MultidimModelStats(BaseModel):
    irt_model: str
    model_id: str
    skill: Vector


def compute_item_accuracy(predictions: LeaderboardPredictions) -> Dict[str, float]:
    accuracies = defaultdict(int)
    n_models = 0
    for model_scores in predictions.scored_predictions.values():
        for item_id, score in model_scores["exact_match"].items():
            accuracies[item_id] += score
        n_models += 1

    for item_id in accuracies:
        accuracies[item_id] = accuracies[item_id] / n_models
    return dict(accuracies)


def load_squad_submissions(dev_predictions: LeaderboardPredictions):
    dev_scores = dev_predictions.model_scores
    id_to_sub = {}
    out = read_json(conf["squad"]["out_v2"])
    for row in out["leaderboard"]:
        if row["submission"]["public"] and row["bundle"]["state"] == "ready":
            submit_info = json.loads(row["bundle"]["metadata"]["description"])
            submit_id = submit_info["submit_id"]
            test_id = submit_info["predict_id"]
            name = row["submission"]["description"]
            id_to_sub[submit_id] = {
                "dev_id": submit_id,
                "test_id": test_id,
                "name": name,
                "test_em": row["scores"]["exact_match"],
                "test_f1": row["scores"]["f1"],
                "dev_em": dev_scores[submit_id]["exact_match"],
                "dev_f1": dev_scores[submit_id]["f1"],
                "created": row["submission"]["created"],
            }
    return id_to_sub


def load_squad_id_to_question():
    squad = SquadV2.parse_file(conf["squad"]["dev_v2"])
    id_to_question = {}
    for page in squad.data:
        title = page.title
        for par in page.paragraphs:
            context = par.context
            for qas in par.qas:
                id_to_question[qas.id] = {
                    "text": qas.question,
                    "answers": "|".join(a.text for a in qas.answers),
                    "answer_position": min(a.answer_start for a in qas.answers)
                    if not qas.is_impossible
                    else -1,
                    "is_impossible": qas.is_impossible,
                    "context": context,
                    "title": title,
                }
    return id_to_question


def str_idx_to_int(str_idx: str, irt_model_type: IrtModelType) -> int:
    """STAN indexing is one based, so convert to zero based, or keep the same for pyro

    Args:
        str_idx (str): string numerical index
        irt_model_type (IrtModelType): pyro vs stan

    Raises:
        ValueError: Only allow certain models

    Returns:
        int: fixed index
    """
    if irt_model_type == IrtModelType.stan_2pl:
        idx = int(str_idx) - 1
    elif irt_model_type in (
        IrtModelType.pyro_2pl,
        IrtModelType.pyro_3pl,
        IrtModelType.pyro_1pl,
        IrtModelType.pyro_md1pl,
    ):
        idx = int(str_idx)
    else:
        raise ValueError(f"Invalid model type: {irt_model_type}")
    return idx


def squad_pred_scores_to_jsonlines(fold: str, metric: str = "exact_match"):
    predictions = LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"][fold])
    output = []
    for subject_id, scores in predictions.scored_predictions.items():
        subject_predictions = {}
        for item_id, value in scores[metric].items():
            subject_predictions[item_id] = {
                "scores": {"exact_match": value},
                "example_id": item_id,
                "submission_id": subject_id,
            }
        # Recover the name after the fact when linking back to test data
        output.append(
            {"submission_id": subject_id, "predictions": subject_predictions, "name": None,}
        )
    return output


class MultidimIrtParsed(BaseModel):
    model_stats: Dict[str, MultidimModelStats]
    example_stats: Dict[str, MultidimExampleStats]
    example_ids: List[str]
    model_ids: List[str]
    irt_model: IrtModelType

    @classmethod
    def from_irt_file(cls: Type[T], file: str) -> T:
        irt_results = MultidimIrtResults.parse_file(file)
        return cls.from_irt_results(irt_results)

    @classmethod
    def from_irt_results(cls: Type[T], irt_results: MultidimIrtResults) -> T:
        # pylint: disable=line-too-long
        example_ids = set()
        irt_2pl_example_stats = {}
        for str_idx, ex_id in irt_results.example_ids.items():
            idx = str_idx_to_int(str_idx, IrtModelType(irt_results.irt_model))
            n_examples = len(irt_results.example_ids)
            if idx < 0 or idx > n_examples - 1:
                raise ValueError(f"Invalid index: {idx}, n_examples={n_examples}")

            example_ids.add(ex_id)
            irt_2pl_example_stats[ex_id] = MultidimExampleStats(
                irt_model=irt_results.irt_model,
                example_id=ex_id,
                diff=irt_results.diff[idx],
                disc=irt_results.disc[idx] if irt_results.disc is not None else None,
                lambda_=irt_results.lambdas[idx] if irt_results.lambdas is not None else None,
            )

        model_ids = set()
        irt_2pl_model_stats = {}
        n_models = len(irt_results.model_ids)
        for str_idx, m_id in irt_results.model_ids.items():
            idx = str_idx_to_int(str_idx, IrtModelType(irt_results.irt_model))

            if idx < 0 or idx > n_models - 1:
                all_ids = [int(str_idx) for str_idx in irt_results.model_ids.keys()]
                max_id = max(all_ids)
                min_id = min(all_ids)
                raise ValueError(
                    f"Invalid index: {idx} model_type={irt_results.irt_model} n_models={n_models} min_id={min_id} max_id={max_id}"
                )
            model_ids.add(m_id)
            irt_2pl_model_stats[m_id] = MultidimModelStats(
                irt_model=irt_results.irt_model, model_id=m_id, skill=irt_results.ability[idx],
            )

        return cls(
            irt_model=IrtModelType(irt_results.irt_model),
            example_stats=irt_2pl_example_stats,
            example_ids=list(example_ids),
            model_stats=irt_2pl_model_stats,
            model_ids=list(model_ids),
        )


class IrtParsed(BaseModel):
    model_stats: Dict[str, ModelStats]
    example_stats: Dict[str, ExampleStats]
    example_ids: List[str]
    model_ids: List[str]
    irt_model: IrtModelType

    @classmethod
    def from_irt_file(cls: Type[T], file: str) -> T:
        irt_results = IrtResults.parse_file(file)
        return cls.from_irt_results(irt_results)

    @classmethod
    def from_multidim_1d_file(cls: Type[T], subject_file: str, item_file: str):
        items = read_jsonlines(item_file)
        item_stats = {}
        all_item_ids = []
        irt_model = "multidim_2pl"
        for it in items:
            item_id = it["submission_id"]
            dim = len(it["item_feat_mu"])
            if dim != 2:
                raise ValueError(f"Invalid dimensions: {dim}")
            disc = it["item_feat_mu"][0]
            diff = it["item_feat_mu"][1]
            item_stats[item_id] = ExampleStats(
                example_id=item_id, diff=diff, irt_model=irt_model, disc=disc
            )
            all_item_ids.append(item_id)

        subjects = read_jsonlines(subject_file)
        subject_stats = {}
        all_subject_ids = []
        for subj in subjects:
            subject_id = subj["submission_id"]
            dim = len(subj["ability_mu"])
            if dim != 1:
                raise ValueError(f"Invalid dimensions: {dim}")
            ability = subj["ability_mu"][0]
            subject_stats[subject_id] = ModelStats(
                model_id=subject_id, skill=ability, irt_model=irt_model
            )
            all_subject_ids.append(subject_id)
        return cls(
            irt_model=IrtModelType(irt_model),
            example_stats=item_stats,
            example_ids=all_item_ids,
            model_stats=subject_stats,
            model_ids=all_subject_ids,
        )

    @classmethod
    def from_irt_results(cls: Type[T], irt_results: IrtResults) -> T:
        # pylint: disable=line-too-long
        example_ids = set()
        irt_2pl_example_stats = {}
        for str_idx, ex_id in irt_results.example_ids.items():
            idx = str_idx_to_int(str_idx, IrtModelType(irt_results.irt_model))
            n_examples = len(irt_results.example_ids)
            if idx < 0 or idx > n_examples - 1:
                raise ValueError(f"Invalid index: {idx}, n_examples={n_examples}")

            example_ids.add(ex_id)
            irt_2pl_example_stats[ex_id] = ExampleStats(
                irt_model=irt_results.irt_model,
                example_id=ex_id,
                diff=irt_results.diff[idx],
                disc=irt_results.disc[idx] if irt_results.disc is not None else None,
                lambda_=irt_results.lambdas[idx] if irt_results.lambdas is not None else None,
            )

        model_ids = set()
        irt_2pl_model_stats = {}
        n_models = len(irt_results.model_ids)
        for str_idx, m_id in irt_results.model_ids.items():
            idx = str_idx_to_int(str_idx, IrtModelType(irt_results.irt_model))

            if idx < 0 or idx > n_models - 1:
                all_ids = [int(str_idx) for str_idx in irt_results.model_ids.keys()]
                max_id = max(all_ids)
                min_id = min(all_ids)
                raise ValueError(
                    f"Invalid index: {idx} model_type={irt_results.irt_model} n_models={n_models} min_id={min_id} max_id={max_id}"
                )
            model_ids.add(m_id)
            irt_2pl_model_stats[m_id] = ModelStats(
                irt_model=irt_results.irt_model, model_id=m_id, skill=irt_results.ability[idx],
            )

        return cls(
            irt_model=IrtModelType(irt_results.irt_model),
            example_stats=irt_2pl_example_stats,
            example_ids=list(example_ids),
            model_stats=irt_2pl_model_stats,
            model_ids=list(model_ids),
        )


class ItemID(BaseModel):
    model_id: str
    example_id: str


class LeaderboardSplits(BaseModel):
    train: List[ItemID]
    test: List[ItemID]


def create_leaderboard_splits(fold: str, seed: int = 42, train_size: float = 0.9):
    # Hard coded here since we'd never have the test data
    with open(conf["squad"]["leaderboard"][fold]) as f:
        items = []
        for line in f:
            submission = json.loads(line)
            submission_id = submission["submission_id"]
            for example_id in submission["predictions"].keys():
                items.append(ItemID(model_id=submission_id, example_id=example_id))
    random.seed(seed)
    train, test = train_test_split(items, train_size=train_size)
    splits = LeaderboardSplits(train=train, test=test)
    write_json(conf["squad"]["leaderboard_splits"][fold], splits)
