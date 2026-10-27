"""Closed-form inverse kinematics for the three-link planar arm.

The wrist sits one last-link length behind the end effector along ``phi``. The
first two links and the base-to-wrist line form a triangle, so the law of
cosines gives the elbow angle, with two branches (elbow angle positive or
negative). The last joint supplies the remaining orientation. Within the
+/-150 degree joint limits no other solutions exist, so this reference is exact
and complete for this arm.
"""

from __future__ import annotations

import numpy as np

from baseline import ORIENTATION_WEIGHT
from data import JOINT_LIMIT, LINK_LENGTHS, fk_numpy

ANGLE_TOLERANCE = 1e-6  # radians; absorbs float rounding at joint bounds and a straight elbow


def _wrap(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _validated(pose: np.ndarray) -> np.ndarray:
    target = np.asarray(pose, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("pose must be a finite array with shape (3,)")
    target = target.copy()
    target[2] = float(_wrap(target[2]))
    return target


def _branches(target: np.ndarray) -> tuple[np.ndarray, bool]:
    """Both elbow branches, wrapped but not limited, and whether the wrist is reachable.

    An unreachable wrist clips the law-of-cosines term, which stretches or folds
    the first two links toward it instead of failing.
    """
    l1, l2, l3 = LINK_LENGTHS
    x, y, phi = target
    wrist_x = x - l3 * np.cos(phi)
    wrist_y = y - l3 * np.sin(phi)
    cos_elbow = (wrist_x**2 + wrist_y**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    reachable = abs(cos_elbow) <= 1.0 + 1e-9
    elbow = np.arccos(np.clip(cos_elbow, -1.0, 1.0))
    q2 = np.array([elbow, -elbow])
    q1 = np.arctan2(wrist_y, wrist_x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q3 = phi - q1 - q2
    return _wrap(np.column_stack((q1, q2, q3))), bool(reachable)


def _legal(branches: np.ndarray) -> np.ndarray:
    return np.all(np.abs(branches) <= JOINT_LIMIT + ANGLE_TOLERANCE, axis=1)


def _cost(q: np.ndarray, target: np.ndarray) -> float:
    """Squared residual with the numerical baseline's 0.10 m/rad orientation weight."""
    pose = fk_numpy(q)
    orientation = float(_wrap(pose[2] - target[2]))
    return float(np.sum((pose[:2] - target[:2]) ** 2) + (ORIENTATION_WEIGHT * orientation) ** 2)


def analytic_solutions(pose: np.ndarray) -> np.ndarray:
    """Return every exact solution inside the joint limits, shape ``(k, 3)`` with k in 0..2."""
    branches, reachable = _branches(_validated(pose))
    if not reachable:
        return np.empty((0, 3))
    if np.allclose(branches[0], branches[1], rtol=0.0, atol=ANGLE_TOLERANCE):
        branches = branches[:1]  # straight elbow: both branches coincide
    return np.clip(branches[_legal(branches)], -JOINT_LIMIT, JOINT_LIMIT)


def analytic_ik(pose: np.ndarray) -> np.ndarray:
    """Solve one pose: the first exact branch inside the joint limits, positive elbow first.

    When none exists (out of reach, or a joint would pass its limit), the clamped
    branch with the smaller residual is a best effort, matching the other
    solvers' contract of always returning bounded joint commands.
    """
    target = _validated(pose)
    branches, reachable = _branches(target)
    legal = _legal(branches)
    if reachable and legal.any():
        return np.clip(branches[int(np.argmax(legal))], -JOINT_LIMIT, JOINT_LIMIT)
    candidates = np.clip(branches, -JOINT_LIMIT, JOINT_LIMIT)
    costs = [_cost(q, target) for q in candidates]
    return candidates[int(np.argmin(costs))]
