# Copyright (c) Facebook, Inc. and its affiliates.
tmux \
    new-session 'caddy run' \; \
    new-window 'poetry run uvicorn --reload leaderboard.www.app:app' \; \
    new-window 'cd frontend && yarn start' \; \
