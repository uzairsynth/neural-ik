"""Evaluate a trained CPU model, a cold-start SciPy numerical baseline and the closed form."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from runtime import configure_plotting
configure_plotting()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from analytic import analytic_ik
from baseline import numerical_ik
from data import DEFAULT_DATA, JOINT_LIMIT, LINK_LENGTHS, ROOT, fk_numpy, load_dataset
from model import load_model, predict_angles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        # Hashes retain exact provenance without publishing host folder names.
        return Path(path).name


def error_metrics(q: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Position accuracy, periodic orientation accuracy, and bound compliance."""
    predicted = fk_numpy(np.asarray(q))
    position_mm = np.linalg.norm(predicted[:, :2] - targets[:, :2], axis=1) * 1000.0
    angle_delta = predicted[:, 2] - targets[:, 2]
    orientation_deg = np.rad2deg(np.abs(np.arctan2(np.sin(angle_delta), np.cos(angle_delta))))
    violations = np.any(np.abs(q) > JOINT_LIMIT + 1e-6, axis=1)
    return {
        "mean_position_mm": float(np.mean(position_mm)),
        "median_position_mm": float(np.median(position_mm)),
        "p95_position_mm": float(np.percentile(position_mm, 95)),
        "mean_orientation_deg": float(np.mean(orientation_deg)),
        "joint_limit_violation_rate": float(np.mean(violations)),
    }


def save_workspace_heatmap(
    targets: np.ndarray, predicted_q: np.ndarray, destination: Path, *, bins: int = 50
) -> None:
    """Plot binned test-set error; empty cells remain masked, not interpolated."""
    position_mm = np.linalg.norm(fk_numpy(predicted_q)[:, :2] - targets[:, :2], axis=1) * 1000.0
    extent = float(np.sum(LINK_LENGTHS))
    edges = np.linspace(-extent, extent, bins + 1)
    counts, _, _ = np.histogram2d(targets[:, 0], targets[:, 1], bins=(edges, edges))
    error_sum, _, _ = np.histogram2d(
        targets[:, 0], targets[:, 1], bins=(edges, edges), weights=position_mm
    )
    means = np.divide(error_sum, counts, out=np.zeros_like(error_sum), where=counts > 0)
    displayed = np.ma.masked_where(counts.T == 0, means.T)
    fig, ax = plt.subplots(figsize=(7.2, 6), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e6e6e6")
    mesh = ax.pcolormesh(edges, edges, displayed, cmap=cmap, shading="flat")
    fig.colorbar(mesh, ax=ax, label="Mean neural position error (mm)")
    ax.set(
        xlabel="Target x (m)",
        ylabel="Target y (m)",
        title="Neural IK error over sampled test targets",
        aspect="equal",
        xlim=(-extent, extent),
        ylim=(-extent, extent),
    )
    ax.text(
        0.01,
        0.01,
        "Grey: no test samples; each cell averages sampled orientations",
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def evaluate(
    *,
    data_path: Path = DEFAULT_DATA,
    checkpoint: Path = ROOT / "artifacts" / "model.pt",
    output_dir: Path = ROOT / "artifacts",
    baseline_samples: int = 500,
    seed: int = 2026,
    threads: int = 1,
) -> dict:
    if baseline_samples < 1:
        raise ValueError("baseline_samples must be positive")
    if threads < 1:
        raise ValueError("threads must be positive")
    started = time.perf_counter()
    torch.set_num_threads(threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(data_path)
    targets = np.asarray(dataset["pose_test"])
    if len(targets) == 0:
        raise ValueError("the test split is empty")
    model = load_model(checkpoint)
    full_q = predict_angles(model, targets)
    full_metrics = error_metrics(full_q, targets)

    count = min(baseline_samples, len(targets))
    indices = np.random.default_rng(seed).choice(len(targets), size=count, replace=False)
    paired_targets = targets[indices]
    # Warm up both methods before the timed loop. Data/model loading, metric
    # calculations, and plotting are deliberately outside per-solve timings.
    for pose in paired_targets[: min(count, 20)]:
        predict_angles(model, pose[None, :])
    numerical_ik(paired_targets[0])
    analytic_ik(paired_targets[0])
    neural_q = np.empty((count, 3), dtype=np.float64)
    numerical_q = np.empty((count, 3), dtype=np.float64)
    analytic_q = np.empty((count, 3), dtype=np.float64)
    neural_times = np.empty(count, dtype=np.float64)
    numerical_times = np.empty(count, dtype=np.float64)
    analytic_times = np.empty(count, dtype=np.float64)
    for index, pose in enumerate(paired_targets):
        before = time.perf_counter()
        neural_q[index] = predict_angles(model, pose[None, :])[0]
        neural_times[index] = time.perf_counter() - before
        before = time.perf_counter()
        numerical_q[index] = numerical_ik(pose)
        numerical_times[index] = time.perf_counter() - before
        before = time.perf_counter()
        analytic_q[index] = analytic_ik(pose)
        analytic_times[index] = time.perf_counter() - before

    neural_metrics = error_metrics(neural_q, paired_targets)
    numerical_metrics = error_metrics(numerical_q, paired_targets)
    analytic_metrics = error_metrics(analytic_q, paired_targets)
    for metrics, durations in (
        (neural_metrics, neural_times),
        (numerical_metrics, numerical_times),
        (analytic_metrics, analytic_times),
    ):
        metrics["mean_time_ms_per_solve"] = float(durations.mean() * 1000.0)
        metrics["median_time_ms_per_solve"] = float(np.median(durations) * 1000.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_workspace_heatmap(targets, full_q, output_dir / "error_heatmap.png")
    metrics = {
        "full_test": {"samples": len(targets), "neural": full_metrics},
        "paired_comparison": {
            "samples": count,
            "selection_seed": seed,
            "neural": neural_metrics,
            "numerical": numerical_metrics,
            "analytic": analytic_metrics,
        },
        "metadata": {
            "seed": seed,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "cpu_count": os.cpu_count(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "dataset": _portable_path(data_path),
            "dataset_sha256": _sha256(data_path),
            "checkpoint": _portable_path(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "baseline": "scipy.optimize.least_squares; analytic Jacobian; up to 3 cold starts; max_nfev=80 per start",
            "analytic": "closed form: wrist point, law-of-cosines elbow, both branches; first legal exact branch, else lower-residual clamped branch",
            "timing": "sequential batch-of-one wall time; guess construction, normalization and output decoding included; warm-up excluded",
            "test_joint_labels_used": False,
            "evaluation_elapsed_seconds": time.perf_counter() - started,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--baseline-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    metrics = evaluate(
        data_path=args.data,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        baseline_samples=args.baseline_samples,
        seed=args.seed,
        threads=args.threads,
    )
    print(f"Evaluation completed in {metrics['metadata']['evaluation_elapsed_seconds']:.2f} seconds.")


if __name__ == "__main__":
    main()
