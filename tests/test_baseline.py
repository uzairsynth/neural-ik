"""Check numerical IK derivatives and attainable target solutions."""

import numpy as np
import pytest

from baseline import _jacobian, _residual, numerical_ik
from data import JOINT_LIMIT, fk_numpy


def test_numerical_jacobian_matches_finite_differences():
    q = np.array([0.25, -0.7, 0.4])
    target = fk_numpy(np.array([-0.2, 0.5, -0.1]))
    step = 1e-6
    finite_difference = np.column_stack(
        [
            (_residual(q + step * direction, target) - _residual(q - step * direction, target))
            / (2.0 * step)
            for direction in np.eye(3)
        ]
    )
    np.testing.assert_allclose(_jacobian(q, target), finite_difference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "reference_q",
    ([0.4, -0.8, 0.3], [-0.7, 1.1, -0.4], [1.0, 0.6, -0.2]),
)
def test_numerical_ik_solves_easy_feasible_targets(reference_q):
    target = fk_numpy(np.asarray(reference_q))
    solution = numerical_ik(target)
    predicted = fk_numpy(solution)
    assert solution.shape == (3,)
    assert np.all(np.abs(solution) <= JOINT_LIMIT + 1e-10)
    assert np.linalg.norm(predicted[:2] - target[:2]) < 1e-5
    orientation_delta = predicted[2] - target[2]
    assert abs(np.arctan2(np.sin(orientation_delta), np.cos(orientation_delta))) < 1e-4
