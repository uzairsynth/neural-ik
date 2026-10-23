"""Reproducible training data for a three-link planar robot arm.

Angles are in radians and distances are in metres.  A pose is ``(x, y, phi)``;
``phi`` is wrapped to the half-open interval [-pi, pi).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LINK_LENGTHS = np.array([0.30, 0.25, 0.15], dtype=np.float64)
JOINT_LIMIT = float(np.deg2rad(150.0))
DEFAULT_DATA = ROOT / "artifacts" / "data.npz"
SPLIT_NAMES = ("train", "val", "test")


def fk_numpy(q: np.ndarray) -> np.ndarray:
    """Return end-effector poses for joint angles with shape ``(..., 3)``.

    Every joint angle is relative to the preceding link.  This function also
    accepts angles outside the training limits, making it useful for numerical
    checks of the kinematics themselves.
    """
    q = np.asarray(q)
    if q.ndim == 0 or q.shape[-1] != 3:
        raise ValueError("Joint angles must have shape (..., 3).")
    cumulative = np.cumsum(q, axis=-1)
    x = np.sum(np.cos(cumulative) * LINK_LENGTHS, axis=-1)
    y = np.sum(np.sin(cumulative) * LINK_LENGTHS, axis=-1)
    phi = (cumulative[..., -1] + np.pi) % (2.0 * np.pi) - np.pi
    return np.stack((x, y, phi), axis=-1)


def generate_dataset(samples: int = 200_000, seed: int = 42) -> dict[str, np.ndarray]:
    """Uniformly sample legal joints and make a seeded 80/10/10 split.

    Splitting is performed by a shuffled index permutation, so examples cannot
    be shared between splits.  For sizes not divisible by ten, the test split
    receives the remainder after flooring the train and validation sizes.
    """
    if not isinstance(samples, (int, np.integer)) or samples < 10:
        raise ValueError("samples must be an integer of at least 10.")
    rng = np.random.default_rng(seed)
    q = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(samples, 3)).astype(np.float32)
    poses = fk_numpy(q).astype(np.float32)
    indices = rng.permutation(samples)
    train_end = int(0.80 * samples)
    val_end = train_end + int(0.10 * samples)
    split_indices = np.split(indices, [train_end, val_end])
    result = {}
    for name, selected in zip(SPLIT_NAMES, split_indices):
        result[f"q_{name}"] = q[selected]
        result[f"pose_{name}"] = poses[selected]
    return result


def save_dataset(
    path: str | Path = DEFAULT_DATA, samples: int = 200_000, seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate, save, and return the dataset as compressed float32 arrays."""
    path = Path(path)
    dataset = generate_dataset(samples=samples, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    # An explicit file handle prevents NumPy silently appending an extension.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **dataset)
    return dataset


def load_dataset(path: str | Path = DEFAULT_DATA) -> dict[str, np.ndarray]:
    """Load a generated dataset without allowing pickled Python objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run python data.py to generate it, "
            "or pass the path of an existing dataset."
        )
    with np.load(path, allow_pickle=False) as archive:
        required = [f"{kind}_{split}" for split in SPLIT_NAMES for kind in ("q", "pose")]
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {', '.join(missing)}")
        return {key: archive[key] for key in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    dataset = save_dataset(path=args.output, samples=args.samples, seed=args.seed)
    counts = ", ".join(f"{name}={len(dataset[f'q_{name}']):,}" for name in SPLIT_NAMES)
    print(f"Saved {args.samples:,} configurations to {args.output.resolve()} ({counts}).")


if __name__ == "__main__":
    main()
