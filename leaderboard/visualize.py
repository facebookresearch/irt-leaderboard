"""Copyright (c) Facebook, Inc. and its affiliates."""
import os
import subprocess
import datetime
from collections import defaultdict
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pedroai.io import read_json

from leaderboard.config import conf
from leaderboard.data import (
    IrtParsed,
    LeaderboardPredictions,
    compute_item_accuracy,
    load_squad_id_to_question,
    load_squad_submissions,
)


## Hack to extend the width of the main pane.
def _max_width_():
    max_width_str = "max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    th {{
        text-align: left;
        font-size: 110%;
    }}
    tr:hover {{
        background-color: #ffff99;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache(allow_output_mutation=True)
def load_cached_squad():
    return load_squad_id_to_question()


@st.cache(allow_output_mutation=True)
def get_predictions():
    return LeaderboardPredictions.parse_file(conf["squad"]["submission_predictions"]["dev"])


@st.cache(allow_output_mutation=True)
def get_item_accuracy():
    data = get_predictions()
    example_accuracy = defaultdict(float)
    example_f1 = defaultdict(float)
    squad = load_cached_squad()
    example_ids = list(squad.keys())
    for model_scores in data.scored_predictions.values():
        for ex_id in example_ids:
            example_accuracy[ex_id] += model_scores["exact_match"][ex_id]
            example_f1[ex_id] += model_scores["f1"][ex_id]
    n_models = len(data.scored_predictions)
    for ex_id, correct in example_accuracy.items():
        example_accuracy[ex_id] = correct / n_models

    for ex_id, correct in example_f1.items():
        example_f1[ex_id] = correct / n_models
    return example_accuracy, example_f1


@st.cache(allow_output_mutation=True)
def load_data(irt_type: str):
    irt_params = get_irt_model(irt_type)
    predictions = get_predictions()
    id_to_question = load_cached_squad()

    id_to_subject = load_squad_submissions(predictions)
    item_accuracy = compute_item_accuracy(predictions)

    model_rows = []
    for model_id in irt_params.model_ids:
        stats = irt_params.model_stats[model_id]
        subject = id_to_subject[model_id]
        model_rows.append(
            {
                "dev_id": model_id,
                "test_id": subject["test_id"],
                "name": subject["name"],
                "skill": stats.skill,
                "dev_em": subject["dev_em"],
                "dev_f1": subject["dev_f1"],
                "test_em": subject["test_em"],
                "test_f1": subject["test_f1"],
                "created": datetime.datetime.fromtimestamp(subject["created"]),
            }
        )
    model_df = pd.DataFrame(model_rows)

    item_rows = []
    for item_id in irt_params.example_ids:
        stats = irt_params.example_stats[item_id]
        question = id_to_question[item_id]
        item_rows.append(
            {
                "item_id": item_id,
                "diff": stats.diff,
                "acc": item_accuracy[item_id],
                "disc": stats.disc,
                "lambda": stats.lambda_,
                "title": question["title"],
                "text": question["text"],
                "is_impossible": question["is_impossible"],
                "context": question["context"],
                "answers": question["answers"],
            }
        )
    item_df = pd.DataFrame(item_rows)

    return model_df, item_df


@st.cache(allow_output_mutation=True)
def get_irt_model(irt_type: str) -> IrtParsed:
    irt_parsed = IrtParsed.from_irt_file(
        Path(conf["irt"]["squad"]["dev"]["pyro"][irt_type]["full"]) / "parameters.json"
    )
    return irt_parsed


def create_irt_dist_chart(irt_results: IrtParsed):
    BASE_SIZE = 300
    rows = []
    squad = load_cached_squad()
    item_em, item_f1 = get_item_accuracy()
    for example in irt_results.example_stats.values():
        item_id = example.example_id
        item = squad[item_id]
        rows.append(
            {
                "disc": example.disc,
                "diff": example.diff,
                "lambda": example.lambda_,
                "avg EM": item_em[item_id],
                "avg F1": item_f1[item_id],
                "item_id": item_id,
                "title": item["title"],
                "question": item["text"],
                "is_impossible": item["is_impossible"],
                "answer": item["answers"],
                "context": item["context"],
            }
        )

    def assign_feas_bin(feas):
        if feas < 0.33:
            return "Low"
        elif feas < 0.66:
            return "Mid"
        else:
            return "High"

    df = pd.DataFrame(rows)
    df["feas_bin"] = df["lambda"].map(assign_feas_bin)
    diff_min = np.floor(df["diff"].min())
    diff_max = np.ceil(df["diff"].max())
    diff_scale = alt.Scale(domain=(diff_min, diff_max))

    disc_min = np.floor(df["disc"].min())
    disc_max = np.ceil(df["disc"].max())
    disc_scale = alt.Scale(domain=(disc_min, disc_max))

    ratio = 1.5
    points = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X("diff", title="Difficulty (𝜃)", scale=diff_scale),
            y=alt.Y("disc", title="Discriminability (𝛾)", scale=disc_scale),
            color=alt.Color(
                "lambda", title="Feasibility (λ)", scale=alt.Scale(scheme="redyellowblue"),
            ),
            tooltip=alt.Tooltip(
                [
                    "item_id",
                    "diff",
                    "disc",
                    "lambda",
                    "avg EM",
                    "avg F1",
                    "title",
                    "question",
                    "is_impossible",
                    "answer",
                ]
            ),
        )
    ).properties(width=ratio * BASE_SIZE, height=ratio * BASE_SIZE)
    top_hist = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("diff", bin=alt.Bin(maxbins=50), stack=None, title="", scale=diff_scale),
            y=alt.Y("count()", stack=True, title=""),
        )
        .properties(height=40, width=ratio * BASE_SIZE)
    )
    right_hist = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X("count()", stack=True, title=""),
            y=alt.Y("disc", bin=alt.Bin(maxbins=50), stack=None, title="", scale=disc_scale),
        )
    ).properties(width=40, height=ratio * BASE_SIZE)
    # points = points + annotations
    chart = top_hist & (points | right_hist)
    chart = chart.configure_concat(spacing=10)

    return chart


@st.cache(allow_output_mutation=True)
def irt_chart(irt_type: str):
    irt_params = get_irt_model(irt_type)
    if irt_type == "1PL":
        return "No Plot for 1PL"
    else:
        return create_irt_dist_chart(irt_params)


ITEM_SORT_COLS = ["diff", "disc", "lambda", "acc"]
STEP = 20

DESC = """
## Instructions

This page has a:
- Table of Subjects
- Table of Items
- Response Inspector
"""

irt_expression = {
    "3PL": r"P_{\text{4PL}}(\text{correct})=\frac{\lambda_j}{1+e^{-\gamma_j (\theta_i - \beta_j)}}",
    "2PL": r"P_{\text{2PL}}(\text{correct})=\frac{1}{1+e^{-\gamma_j (\theta_i - \beta_j)}}",
    "1PL": r"P_{\text{1PL}}(\text{correct})=\frac{1}{1+e^{- (\theta_i - \beta_j)}}",
}


def prob_4pl(*, theta: float, beta: float, gamma: float, c: float, lambda_: float):
    return c + (lambda_ - c) / (1 + np.exp(-gamma * (theta - beta)))


def prob_2pl(*, theta: float, beta: float, gamma: float):
    return 1 / (1 + np.exp(-gamma * (theta - beta)))


def prob_1pl(*, theta: float, beta: float):
    return 1 / (1 + np.exp(-(theta - beta)))


@st.cache(allow_output_mutation=True)
def load_subject_guesses(subject_id: str):
    return read_json(f"data/squad/submissions/{subject_id}.json")


@st.cache(allow_output_mutation=True)
def load_subject_probs(irt_model: str, subject_id: str):
    irt_params = get_irt_model(irt_model)
    ability = irt_params.model_stats[subject_id].skill
    item_to_prob = {}
    for item_id in irt_params.example_ids:
        difficulty = irt_params.example_stats[item_id].diff
        disc = irt_params.example_stats[item_id].disc
        lambda_ = irt_params.example_stats[item_id].lambda_
        if irt_model == "3PL":
            prob = prob_4pl(theta=ability, beta=difficulty, gamma=disc, c=0, lambda_=lambda_)
        elif irt_model == "2PL":
            prob = prob_2pl(theta=ability, beta=difficulty, gamma=disc)
        elif irt_model == "1PL":
            prob = prob_1pl(theta=ability, beta=difficulty)
        else:
            raise ValueError("Invalid IRT Model")
        item_to_prob[item_id] = prob
    return item_to_prob


@st.cache
def load_subject_answer(subject_id: str, item_id: str):
    guesses = load_subject_guesses(subject_id)
    return guesses[item_id]


@st.cache
def load_single_subject_df(irt_model: str, subject_id: str):
    id_to_question = load_cached_squad()
    irt_params = get_irt_model(irt_model)
    predictions = get_predictions()
    subject_probs = load_subject_probs(irt_model, subject_id)
    # subject_stats = irt_params.model_stats[subject_id]
    # ability = subject_stats.skill
    pred_scores = predictions.scored_predictions[subject_id]["exact_match"]
    guesses = load_subject_guesses(subject_id)
    rows = []
    for item_id in irt_params.example_ids:
        # item_stats = irt_params.example_stats[item_id]
        question = id_to_question[item_id]
        rows.append(
            {
                "item_id": item_id,
                "irt_prob": subject_probs[item_id],
                "score": pred_scores[item_id],
                "abs(irt_prob - score)": np.abs(pred_scores[item_id] - subject_probs[item_id]),
                # "ability": ability,
                # "diff": item_stats.diff,
                # "disc": item_stats.disc,
                # "feasibility": item_stats.lambda_,
                "title": question["title"],
                "question": question["text"],
                "is_impossible": question["is_impossible"],
                "answer": question["answers"],
                "guess": guesses[item_id],
                "context": question["context"],
            }
        )

    df = pd.DataFrame(rows)
    return df


def download_data(force: bool = False):
    should_download = bool(os.environ.get('DOWNLOAD_DATA'))
    files_exist = os.path.exists('data/irt/squad/dev/pyro/3PL_full/parameters.json')

    if force or (should_download and not files_exist):
        print("Downloading data files")
        subprocess.run('wget https://obj.umiacs.umd.edu/acl2021-leaderboard/leaderboard-data-only-irt.tar.gz')
        subprocess.run('tar xzvf leaderboard-data-only-irt.tar.gz')


def main():
    st.header("Welcome to the SQuAD 2.0 IRT Leaderboard!")
    st.markdown(
        """
    On this page, you can inspect the IRT parameters
    and predictions of each 1D IRT model in our paper.
    In order, we show the list of subjects, items,
    and a subject-item response inspector.

    On the left column, there are controls for each table
    which let you sort by:
    * Each IRT parameter
    * SQuAD evaluation metrics
    * In ascending or descending order
    * To view the next page, click the -/+ buttons
    """
    )
    st.sidebar.title("IRT Viewer")
    st.sidebar.markdown(DESC)
    st.sidebar.subheader("IRT Model Selection")
    irt_model = st.sidebar.selectbox("IRT Model", ["3PL", "2PL", "1PL"])
    st.sidebar.latex(r"\theta=\text{ability/skill}, \beta=\text{difficulty}")
    st.sidebar.latex(r"\gamma=\text{discriminability}, \lambda=\text{feasibility}")
    st.sidebar.latex(irt_expression[irt_model])
    model_df, item_df = load_data(irt_model)

    st.sidebar.subheader("Subject Filters")
    subject_sort_col_primary = st.sidebar.selectbox(
        "Sort By (Subject)", ["skill", "test_em", "test_f1", "dev_em", "dev_f1", "created"],
    )
    subject_ascending = st.sidebar.checkbox("Ascending (Subject)", value=False)
    subject_offset = st.sidebar.number_input(
        "Subject Offset (Size: %d)" % len(model_df),
        min_value=0,
        max_value=int(len(item_df)) - STEP,
        value=0,
        step=STEP,
    )

    st.sidebar.subheader("Item Filters")
    item_sort_col_primary = st.sidebar.selectbox("Primary Sort By (Item)", ITEM_SORT_COLS)
    item_ascending = st.sidebar.checkbox("Ascending (Item)", value=True)
    item_abs = st.sidebar.checkbox("Absolute Value", value=False)
    if item_abs:
        item_sort_key = np.abs
    else:
        item_sort_key = None
    item_offset = st.sidebar.number_input(
        "Item Offset (Size: %d)" % len(item_df),
        min_value=0,
        max_value=int(len(item_df)) - STEP,
        value=0,
        step=STEP,
    )

    st.header("Subjects")
    st.subheader("This table shows all the subjects (SQuAD Models)")
    st.table(
        model_df.sort_values(subject_sort_col_primary, ascending=subject_ascending).iloc[
            subject_offset : subject_offset + STEP
        ]
    )

    st.header("Items")
    st.subheader("The table, chart, and inspector show each item (question)")
    id_to_question = load_cached_squad()
    all_item_ids = list(id_to_question)
    item_accuracy, item_f1 = get_item_accuracy()
    left, right = st.beta_columns(2)
    irt_params = get_irt_model(irt_model)
    with left:
        st.subheader("Hover over points to see more about it")
        st.write(irt_chart(irt_model))
    with right:
        st.subheader("Choose an Item ID to View")
        inspect_item_id = st.selectbox("Choose an Item ID:", all_item_ids)
        chosen_item = id_to_question[inspect_item_id].copy()
        chosen_item["mean_exact_match"] = item_accuracy[inspect_item_id]
        chosen_item["mean_f1"] = item_f1[inspect_item_id]
        chosen_item["difficulty"] = irt_params.example_stats[inspect_item_id].diff
        chosen_item["discriminability"] = irt_params.example_stats[inspect_item_id].disc
        chosen_item["feasibility"] = irt_params.example_stats[inspect_item_id].lambda_
        st.write(chosen_item)
    st.table(
        item_df.sort_values(
            [item_sort_col_primary], ascending=item_ascending, key=item_sort_key
        ).iloc[item_offset : item_offset + STEP]
    )

    predictions = get_predictions()
    all_subject_ids = model_df.dev_id.tolist()
    st.sidebar.subheader("Subject-Item Response Inspector")
    st.header("Subject-Item Response Inspector")
    st.subheader("Here you can inspect IRT prediction for item-subject pairs")
    item_id = st.sidebar.selectbox("Item ID", all_item_ids)
    subject_id = st.sidebar.selectbox("Subject Dev ID", all_subject_ids)
    st.write(id_to_question[item_id])

    subject_probs = load_subject_probs(irt_model, subject_id)
    prob = subject_probs[item_id]

    correct = predictions.scored_predictions[subject_id]["exact_match"][item_id]
    guess = load_subject_answer(subject_id, item_id)
    if guess == "":
        guess = "No Answer: Subject Abstains"

    st.write(
        {
            "subject_id": subject_id,
            "item_id": item_id,
            "ability": irt_params.model_stats[subject_id].skill,
            "difficulty": irt_params.example_stats[item_id].diff,
            "discriminability": irt_params.example_stats[item_id].disc,
            "feasibility": irt_params.example_stats[item_id].lambda_,
            "p_correct": prob,
            "subject_guess": guess,
            "subject_answer_correct": correct,
        }
    )
    st.sidebar.subheader("Subject-Item Filters")
    subject_item_sort_col_primary = st.sidebar.selectbox(
        "Sort By (Subject-Item)",
        [
            "irt_prob",
            # "ability", "diff", "disc", "feasibility",
            "abs(irt_prob - score)",
        ],
    )
    subject_item_ascending = st.sidebar.checkbox("Ascending", value=True)
    subject_item_abs = st.sidebar.checkbox("Absolute Value (Subject-Item)", value=False)
    if subject_item_abs:
        subject_item_sort_key = np.abs
    else:
        subject_item_sort_key = None
    subject_item_df = load_single_subject_df(irt_model, subject_id)
    subject_item_offset = st.sidebar.number_input(
        "Offset (Size: %d) Subject-Item" % len(subject_item_df),
        min_value=0,
        max_value=int(len(subject_item_df)) - STEP,
        value=0,
        step=STEP,
    )
    st.header("Subject-Item Responses")
    st.table(
        subject_item_df.sort_values(
            [subject_item_sort_col_primary],
            ascending=subject_item_ascending,
            key=subject_item_sort_key,
        ).iloc[subject_item_offset : subject_item_offset + STEP]
    )


if __name__ == "__main__":
    _max_width_()
    main()
