"""Copyright (c) Facebook, Inc. and its affiliates."""
import numpy as np

from leaderboard.stats import kuder_richardson_formula_20


def test_kuder_richardson():
    student_item_matrix = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert student_item_matrix.shape[0] == 12
    assert student_item_matrix.shape[1] == 11
    reliability, _ = kuder_richardson_formula_20(student_item_matrix)
    assert np.isclose(reliability, 0.738, atol=0.0, rtol=1e-4)
