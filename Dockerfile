# Copyright (c) Facebook, Inc. and its affiliates.
FROM python:3.7

RUN mkdir -p /code/leaderboard
RUN mkdir /code/leaderboard/data
WORKDIR /code/leaderboard
VOLUME /code/leaderboard/data
VOLUME /code/leaderboard/static-data

RUN pip install poetry==1.1.4
COPY pyproject.toml poetry.lock /code/leaderboard/
RUN poetry export --without-hashes -f requirements.txt > reqs.txt \
    && pip install -r reqs.txt

COPY . /code/leaderboard
CMD ["uvicorn", "--port", "8000", "--host", "0.0.0.0", "--workers", "1", "leaderboard.www.app:app"]