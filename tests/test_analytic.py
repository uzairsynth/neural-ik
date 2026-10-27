"""Check the closed-form solver against forward kinematics and the joint limits."""

import numpy as np
import pytest

from analytic import analytic_ik, analytic_solutions
from data import JOINT_LIMIT, fk_numpy


def _pose_error(q, target):
    predicted = fk_numpy(q)
    delta = predicted[2] - target[2]
    return np.linalg.norm(predicted[:2] - target[:2]), abs(np.arctan2(np.sin(delta), np.cos(delta)))


def test_closed_form_recovers_random_legal_targets():
    q = np.random.default_rng(11).uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(500, 3))
    for reference, target in zip(q, fk_numpy(q)):
        solution = analytic_ik(target)
        position, orientation = _pose_error(solution, target)
        assert np.all(np.abs(solution) <= JOINT_LIMIT)
        assert position < 1e-9
        assert orientation < 1e-9
        solutions = analytic_solutions(target)
        assert any(np.allclose(candidate, reference, atol=1e-6) for candidate in solutions)


def test_closed_form_returns_both_elbow_branches():
    target = fk_numpy(np.deg2rad([35.0, -65.0, 45.0]))
    solutions = analytic_solutions(target)
    assert solutions.shape == (2, 3)
    assert np.sign(solutions[0, 1]) == -np.sign(solutions[1, 1])
    for solution in solutions:
        assert max(_pose_error(solution, target)) < 1e-9


def test_straight_arm_has_one_solution():
    target = np.array([0.70, 0.0, 0.0])
    np.testing.assert_allclose(analytic_solutions(target), [[0.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(analytic_ik(target), [0.0, 0.0, 0.0], atol=1e-6)


def test_unreachable_target_gets_bounded_best_effort():
    target = np.array([1.0, 0.0, 0.0])
    assert analytic_solutions(target).shape == (0, 3)
    solution = analytic_ik(target)
    assert np.all(np.abs(solution) <= JOINT_LIMIT)
    np.testing.assert_allclose(solution, [0.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("pose", [[0.4, 0.2], [0.4, 0.2, np.nan], np.zeros((2, 3))])
def test_closed_form_rejects_invalid_poses(pose):
    with pytest.raises(ValueError, match="shape"):
        analytic_ik(pose)
