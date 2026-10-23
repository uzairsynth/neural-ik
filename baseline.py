"""Deterministic, bounded numerical inverse kinematics using SciPy.

Every call starts from the same target-dependent guesses. Neither reference joint
angles nor a neural prediction is used to initialize the numerical solver.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from data import JOINT_LIMIT, LINK_LENGTHS

ORIENTATION_WEIGHT = 0.10  # metres per radian in the optimization residual


def _wrap(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _residual(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    theta = np.cumsum(q)
    position = np.array(
        [np.sum(LINK_LENGTHS * np.cos(theta)), np.sum(LINK_LENGTHS * np.sin(theta))]
    )
    return np.array(
        [
            position[0] - target[0],
            position[1] - target[1],
            ORIENTATION_WEIGHT * float(_wrap(theta[-1] - target[2])),
        ]
    )


def _jacobian(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of the position and wrapped-orientation residual."""
    del target
    theta = np.cumsum(q)
    dx = -np.cumsum((LINK_LENGTHS * np.sin(theta))[::-1])[::-1]
    dy = np.cumsum((LINK_LENGTHS * np.cos(theta))[::-1])[::-1]
    return np.vstack((dx, dy, np.full(3, ORIENTATION_WEIGHT)))


def numerical_ik(pose: np.ndarray, *, max_nfev: int = 80) -> np.ndarray:
    """Solve one pose with up to three cold starts and joint bounds.

    The first two guesses bend in opposite directions and point approximately
    toward the target. The last guess is the zero configuration. All guess
    construction, optimization, and restart work is included in a timed call.
    A reachable pose is not guaranteed to be found by this local baseline.
    """
    target = np.asarray(pose, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("pose must be a finite array with shape (3,)")
    target = target.copy()
    target[2] = float(_wrap(target[2]))
    heading = np.arctan2(target[1], target[0])
    guesses = []
    for bend in (1.2, -1.2):
        shoulder = float(_wrap(heading - 0.5 * bend))
        wrist = float(_wrap(target[2] - shoulder - bend))
        guesses.append(np.array([shoulder, bend, wrist]))
    guesses.append(np.zeros(3, dtype=np.float64))

    best_q = guesses[-1]
    best_cost = np.inf
    for guess in guesses:
        result = least_squares(
            _residual,
            np.clip(guess, -JOINT_LIMIT + 1e-8, JOINT_LIMIT - 1e-8),
            jac=_jacobian,
            bounds=(-JOINT_LIMIT, JOINT_LIMIT),
            args=(target,),
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-10,
        )
        residual = _residual(result.x, target)
        cost = float(residual @ residual)
        if cost < best_cost:
            best_q = result.x.copy()
            best_cost = cost
        if np.linalg.norm(residual[:2]) < 1e-6 and abs(residual[2]) < 1e-6:
            break
    return np.asarray(best_q, dtype=np.float64)
