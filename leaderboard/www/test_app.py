"""Copyright (c) Facebook, Inc. and its affiliates."""
from fastapi.testclient import TestClient
from fastapi_cache import caches
from fastapi_cache.backends.memory import CACHE_KEY, InMemoryCacheBackend

from leaderboard.www.app import app, initialize_cache

# These are normally done with a startup event, but doesn't look like TestClient supports that
initialize_cache()
mem_cache = InMemoryCacheBackend()
caches.set(CACHE_KEY, mem_cache)
client = TestClient(app)


def test_submissions():
    response = client.get("/api/1.0/submissions")
    assert response.status_code == 200
    parsed = response.json()["submissions"]
    assert len(parsed) != 0
    submission = parsed[0]
    assert "created" in submission
    assert "name" in submission
    assert "dev_scores" in submission
    assert "test_scores" in submission
    assert "submission_id" in submission
    assert "submitter" in submission
    assert "task" in submission


def test_plot_submissions():
    response = client.get("/api/1.0/submissions/plot")
    assert response.status_code == 200


def test_stats_plot():
    response = client.get("/api/1.0/stats/plot")
    assert response.status_code == 200


def test_examples():
    response = client.get("/api/1.0/examples")
    assert response.status_code == 200


def test_examples_irt():
    response = client.get("/api/1.0/examples/plot_irt")
    assert response.status_code == 200


def test_plot_metrics():
    response = client.get("/api/1.0/metrics/plot")
    assert response.status_code == 200
