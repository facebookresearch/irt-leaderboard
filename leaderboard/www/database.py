"""Copyright (c) Facebook, Inc. and its affiliates."""
import json
from contextlib import contextmanager
from typing import Iterable, Optional

from pedroai.io import read_json
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, selectinload, sessionmaker
from tqdm import tqdm

from leaderboard import stats
from leaderboard.analysis.squad import load_squad
from leaderboard.config import DATA_ROOT, conf
from leaderboard.data import LeaderboardPredictions, LeaderboardSubmissions
from leaderboard.log import get_logger

log = get_logger(__name__)

Base = declarative_base()
engine = create_engine(
    f"sqlite:///{DATA_ROOT / conf['db']}", connect_args={"check_same_thread": False}
)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> SessionLocal:
    session = SessionLocal()
    yield session
    session.close()


class Submission(Base):
    __tablename__ = "submissions"
    dbid = Column(Integer, primary_key=True)
    submission_id = Column(String, nullable=False)
    created = Column(DateTime)
    name = Column(String)
    submitter = Column(String)
    dev_scores = Column(JSON)
    test_scores = Column(JSON)
    task = Column(String, nullable=False)
    predictions = relationship("Prediction")

    def to_dict(self):
        return {
            "submission_id": self.submission_id,
            "dbid": self.dbid,
            "created": self.created,
            "name": self.name,
            "submitter": self.submitter,
            "dev_scores": self.dev_scores,
            "test_scores": self.test_scores,
            "task": self.task,
        }


class Example(Base):
    __tablename__ = "examples"
    dbid = Column(Integer, primary_key=True)
    example_id = Column(String)
    task = Column(String, nullable=False)
    data = Column(JSON)
    predictions = relationship("Prediction")


class Prediction(Base):
    __tablename__ = "predictions"
    dbid = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.dbid"), index=True)
    example_id = Column(String, ForeignKey("examples.dbid"))
    example = relationship("Example", back_populates="predictions")
    submission = relationship("Submission", back_populates="predictions")
    task = Column(String, nullable=False)
    scores = Column(JSON)
    output = Column(JSON)


class StatisticalTest(Base):
    __tablename__ = "statistical_tests"
    dbid = Column(Integer, primary_key=True)
    model_a_id = Column(String, ForeignKey("submissions.submission_id"), nullable=False, index=True)
    model_b_id = Column(String, ForeignKey("submissions.submission_id"), nullable=False, index=True)
    model_a = relationship("Submission", foreign_keys=[model_a_id])
    model_b = relationship("Submission", foreign_keys=[model_b_id])
    key = Column(String, nullable=False, index=True)
    score_a = Column(Float, nullable=False)
    score_b = Column(Float, nullable=False)
    statistic = Column(Float, nullable=True)
    pvalue = Column(Float, nullable=True)
    test = Column(String, nullable=False, index=True)
    max_score = Column(Float, nullable=False)
    min_score = Column(Float, nullable=False)
    diff = Column(Float, nullable=False)
    fold = Column(String, nullable=False, index=True)
    metric = Column(String, nullable=False, index=True)

    def to_dict(self):
        return stats.PairedStats(
            model_a=self.model_a_id,
            model_b=self.model_b_id,
            key=self.key,
            score_a=self.score_a,
            score_b=self.score_b,
            statistic=self.statistic,
            pvalue=self.pvalue,
            test=self.test,
            max_score=self.max_score,
            min_score=self.min_score,
            diff=self.diff,
            fold=self.fold,
            metric=self.metric,
        ).dict()


def build_db(limit_submissions: Optional[int] = None, skip_tests: bool = False):
    # pylint: disable=bare-except
    try:
        Base.metadata.drop_all(bind=engine)
    except:
        pass
    Base.metadata.create_all(bind=engine)
    build_submissions(limit=limit_submissions)
    # Don't run this on CI
    if limit_submissions is None:
        if not skip_tests:
            log.info("Building stat tests")
            build_tests()


def build_submissions(limit=None):
    log.info("Loading submission metadata")
    leaderboard = LeaderboardSubmissions(
        **read_json(DATA_ROOT / conf["squad"]["submission_metadata"])
    )
    _, question_map = load_squad()
    leaderboard_predictions = LeaderboardPredictions.parse_file(
        DATA_ROOT / conf["squad"]["submission_predictions"]["dev"]
    )
    scores = leaderboard_predictions.scored_predictions
    examples = {}
    with get_db_context() as db:
        for qid, ex in question_map.items():
            db_example = Example(task="squad", data=ex, example_id=qid)
            examples[qid] = db_example
            db.add(db_example)
        n = 0
        for s in tqdm(leaderboard.submissions):
            dev_scores = leaderboard_predictions.model_scores[s.submit_id]
            submission = Submission(
                created=s.created,
                name=s.name,
                submitter=s.submitter,
                dev_scores=dev_scores,
                test_scores=s.scores,
                submission_id=s.submit_id,
                task="squad",
            )
            submission_predictions = read_json(f"data/squad/submissions/{s.submit_id}.json")
            metric_to_scores = scores[s.submit_id]
            predictions = []
            for qid, pred in submission_predictions.items():
                qid_scores = {}
                for metric_name, qids_to_scores in metric_to_scores.items():
                    qid_scores[metric_name] = qids_to_scores[qid]
                predictions.append(
                    Prediction(
                        submission=submission,
                        example=examples[qid],
                        task="squad",
                        scores=qid_scores,
                        output=pred,
                    )
                )
            submission.predictions = predictions
            db.add(submission)
            if limit is not None and n > limit:
                break
            n += 1
        log.info("Committing")
        db.commit()


def build_tests():
    log.info("Loading Statistical tests")
    with get_db_context() as db:
        for t in stats.TESTS:
            results = read_json(DATA_ROOT / f"data/stats/full/{t}.json")["results"]
            for row in tqdm(results):
                stat_test = stats.PairedStats(**row)
                db.add(
                    StatisticalTest(
                        model_a_id=stat_test.model_a,
                        model_b_id=stat_test.model_b,
                        key=stat_test.key,
                        score_a=stat_test.score_a,
                        score_b=stat_test.score_b,
                        statistic=stat_test.statistic,
                        pvalue=stat_test.pvalue,
                        test=stat_test.test,
                        max_score=stat_test.max_score,
                        min_score=stat_test.min_score,
                        diff=stat_test.diff,
                        fold=stat_test.fold,
                        metric=stat_test.metric,
                    )
                )
            log.info("Committing Test: %s", t)
            db.commit()


def export_submissions():
    with get_db_context() as db:
        db_submissions: Iterable[Submission] = (
            db.query(Submission).options(selectinload(Submission.predictions)).all()
        )
        submissions = []
        n_preds_per_submission = set()
        for sub in tqdm(db_submissions):
            predictions = {}
            for pred in sub.predictions:
                if pred.example_id in predictions:
                    raise ValueError(f"Prediction already exists: {pred.example_id}")
                predictions[pred.example_id] = {
                    "scores": json.loads(pred.scores),
                    "example_id": pred.example_id,
                    "submission_id": sub.bundle_id,
                }
            n_preds_per_submission.add(len(predictions))

            submissions.append(
                {"predictions": predictions, "submission_id": sub.bundle_id, "name": sub.name,}
            )
        if len(n_preds_per_submission) != 1:
            raise ValueError(f"Mismatching number of preds: {len(n_preds_per_submission)}")
        return submissions
