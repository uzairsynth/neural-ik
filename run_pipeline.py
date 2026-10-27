"""Regenerate the dataset, train, benchmark, and refresh README measurements."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def refresh_readme():
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    paired = metrics["paired_comparison"]
    full = metrics["full_test"]["neural"]
    lines = [
        f"Measured on **{paired['samples']:,} identical test targets** (selection seed {paired['selection_seed']}); "
        "all times are warmed-up, sequential CPU solves with batch size one.",
        "",
        "| Solver | Mean position (mm) | Median (mm) | 95th percentile (mm) | Mean orientation (°) | Mean solve (ms) |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    solvers = [(key, name) for key, name in (("neural", "Neural MLP"), ("numerical", "SciPy least squares"),
                                             ("analytic", "Closed form (exact)")) if key in paired]
    for key, name in solvers:
        row = paired[key]
        lines.append(f"| {name} | {row['mean_position_mm']:.3f} | {row['median_position_mm']:.3f} | "
                     f"{row['p95_position_mm']:.3f} | {row['mean_orientation_deg']:.3f} | "
                     f"{row['mean_time_ms_per_solve']:.3f} |")
    lines += ["", f"On **all {metrics['full_test']['samples']:,} test samples**, the neural mean / median / 95th-percentile "
              f"position errors are **{full['mean_position_mm']:.3f} / {full['median_position_mm']:.3f} / "
              f"{full['p95_position_mm']:.3f} mm**; mean orientation error is {full['mean_orientation_deg']:.3f}°. "
              "Joint-limit violations: " + ", ".join(
                  f"{name} {100 * paired[key]['joint_limit_violation_rate']:.1f}%" for key, name in solvers) + "."]
    metadata = metrics["metadata"]
    lines += ["", f"Evaluation environment: Python {metadata['python']}, PyTorch {metadata['torch']}, "
              f"NumPy {metadata['numpy']}, SciPy {metadata['scipy']}; "
              f"{metadata['torch_threads']} PyTorch inference thread(s). "
              f"CPU: {metadata['processor'] or 'not reported'}."]
    summary_path = ROOT / "artifacts" / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines += ["", f"Checkpoint training function: {summary['training_seconds']:.1f} seconds, {summary['epochs_completed']} epochs, "
                  f"best validation checkpoint at epoch {summary['best_epoch']}; stopped by {summary['stop_reason']}. "
                  f"Evaluation function: {metadata['evaluation_elapsed_seconds']:.1f} seconds. "
                  "These function timings exclude process startup and imports; the full pipeline time below includes them."]
    runtime_path = ROOT / "artifacts" / "pipeline_summary.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        host = f"CPU: {runtime['processor']}" if runtime.get("processor") else "CPU"
        lines += ["", f"Last complete data → training → evaluation → tests run: "
                  f"**{runtime['total_seconds']:.1f} seconds** on {host} (dependencies already installed)."]
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            readme.write_text(before + start + "\n" + "\n".join(lines) + "\n" + end + after, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-readme-only", action="store_true")
    parser.add_argument("--train-seconds", type=float, default=360)
    args = parser.parse_args()
    if args.refresh_readme_only:
        refresh_readme()
        return
    started = time.perf_counter()
    commands = [
        ["data.py"],
        ["train.py", "--max-seconds", str(args.train_seconds)],
        ["evaluate.py"],
        ["-m", "pytest", "-q"],
    ]
    stages = {}
    for command in commands:
        before = time.perf_counter()
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
        stages[" ".join(command)] = time.perf_counter() - before
    total = time.perf_counter() - started
    summary = {"total_seconds": total, "within_ten_minutes": total < 600, "stages_seconds": stages,
               "processor": platform.processor()}
    (ROOT / "artifacts" / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    refresh_readme()
    print(f"Pipeline finished in {total:.1f}s; README measurements refreshed.")
    if total >= 600:
        print("This host exceeded 10 minutes; reduce --train-seconds to reserve more runtime headroom.")


if __name__ == "__main__":
    main()
