"""Interactive CPU comparison of the learned and numerical IK solvers."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from runtime import configure_plotting
configure_plotting()
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from baseline import numerical_ik
from data import JOINT_LIMIT, LINK_LENGTHS, fk_numpy
from model import load_model, predict_angles

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "model.pt"


@lru_cache(maxsize=2)
def _cached_model(checkpoint: str):
    """Keep loading and initial setup outside the displayed inference time."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {path}. Run python train.py first."
        )
    return load_model(checkpoint=path)


def arm_points(angles: np.ndarray) -> np.ndarray:
    """Return the base and three joint/end-effector positions, in metres."""
    headings = np.cumsum(np.asarray(angles, dtype=float))
    steps = np.asarray(LINK_LENGTHS)[:, None] * np.column_stack(
        (np.cos(headings), np.sin(headings))
    )
    return np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))


def _pose_errors(angles: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual = fk_numpy(angles)
    position_mm = float(np.linalg.norm(actual[:2] - target[:2]) * 1000)
    delta_phi = np.arctan2(np.sin(actual[2] - target[2]), np.cos(actual[2] - target[2]))
    return position_mm, float(np.abs(np.rad2deg(delta_phi)))


def draw_solutions(target: np.ndarray, solutions: dict[str, np.ndarray]) -> Figure:
    """Build an unmanaged Figure; repeated calls do not accumulate pyplot windows."""
    figure = Figure(figsize=(7, 6), layout="constrained")
    axis = figure.subplots()
    reach = float(np.sum(LINK_LENGTHS))
    axis.add_patch(Circle((0, 0), reach, color="#e7edf4", alpha=0.65))
    axis.add_patch(Circle((0, 0), reach, fill=False, color="#a5b5c7", linestyle=":"))
    colors = {"Neural": "#1967d2", "Numerical": "#e47918"}
    for name, angles in solutions.items():
        points = arm_points(angles)
        color = colors[name]
        axis.plot(
            points[:, 0], points[:, 1], "o-", color=color,
            linewidth=3, markersize=6, alpha=0.85, label=name,
        )
        heading = float(np.sum(angles))
        axis.annotate(
            "", xy=points[-1] + 0.08 * np.array([np.cos(heading), np.sin(heading)]),
            xytext=points[-1], arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
        )
    axis.scatter(*target[:2], marker="x", s=95, linewidths=2.2, color="#172b4d", label="Target", zorder=5)
    axis.annotate(
        "", xy=target[:2] + 0.10 * np.array([np.cos(target[2]), np.sin(target[2])]),
        xytext=target[:2], arrowprops={"arrowstyle": "->", "color": "#172b4d", "lw": 2},
    )
    axis.scatter(0, 0, s=60, color="#172b4d", zorder=5)
    axis.set(xlim=(-0.83, 0.83), ylim=(-0.83, 0.83), xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal")
    axis.set_title("Arm configurations and end-effector directions")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    return figure


def solve_target(
    x: float,
    y: float,
    phi_degrees: float,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> tuple[Figure, str]:
    """Solve one target, returning a plot and a readable comparison table.

    Timing includes one solver call after loading the model, not rendering.
    It is an interactive measurement; evaluate.py provides the benchmark.
    """
    target = np.array([x, y, np.deg2rad(phi_degrees)], dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Target values must be finite.")
    model = _cached_model(str(Path(checkpoint).resolve()))
    start = perf_counter()
    neural = np.asarray(predict_angles(model, target[None, :])[0], dtype=float)
    neural_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    numerical = np.asarray(numerical_ik(target), dtype=float)
    numerical_ms = (perf_counter() - start) * 1000
    solutions = {"Neural": neural, "Numerical": numerical}
    rows = []
    errors = {}
    for name, elapsed_ms in (("Neural", neural_ms), ("Numerical", numerical_ms)):
        angles = solutions[name]
        position_mm, orientation_deg = _pose_errors(angles, target)
        errors[name] = (position_mm, orientation_deg)
        within_limits = bool(np.all(np.abs(angles) <= float(JOINT_LIMIT) + 1e-6))
        joint_text = ", ".join(f"{angle:.1f}" for angle in np.rad2deg(angles))
        rows.append(
            f"| {name} | {position_mm:.2f} | {orientation_deg:.2f} | {elapsed_ms:.3f} | "
            f"{joint_text} | {'Yes' if within_limits else 'No'} |"
        )
    if np.linalg.norm(target[:2]) > np.sum(LINK_LENGTHS) + 1e-9:
        status = "**Outside arm reach:** the requested position is farther than 0.70 m from the base."
    elif errors["Numerical"][0] > 1.0 or errors["Numerical"][1] > 1.0:
        status = (
            "**Numerical solver did not meet the 1 mm / 1° target tolerance.** "
            "The pose may violate joint/orientation constraints, or the optimizer may have found a local minimum."
        )
    else:
        status = "**Numerical target reached** within 1 mm and 1°. Compare the neural residual below."
    report = "\n".join([
        status, "", "| Solver | Position (mm) | Orientation (°) | Solve (ms) | Joints q1, q2, q3 (°) | Within ±150° |",
        "|:--|--:|--:|--:|:--|:--:|", *rows, "",
        "Arrows show end-effector direction. Timings exclude checkpoint loading and plotting; "
        "use `evaluate.py` for aggregate benchmark results.",
    ])
    return draw_solutions(target, solutions), report


def create_app(checkpoint: str | Path = DEFAULT_CHECKPOINT):
    """Construct the demo without starting a server (also useful for smoke tests)."""
    import gradio as gr

    def solve(x, y, phi_degrees):
        try:
            return solve_target(x, y, phi_degrees, checkpoint)
        except FileNotFoundError as exc:
            raise gr.Error(str(exc)) from exc

    initial = fk_numpy(np.deg2rad(np.array([35.0, -65.0, 45.0])))
    with gr.Blocks(title="neural-ik · Planar arm") as demo:
        gr.Markdown(
            "# neural-ik\n"
            "Compare a PyTorch inverse-kinematics model with a bounded numerical solver. "
            "The arm has 0.30, 0.25, and 0.15 m links, with each joint limited to ±150°. "
            "Set a target pose and select **Solve target**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                x = gr.Slider(-0.70, 0.70, value=float(initial[0]), step=0.005, label="Target x (m)")
                y = gr.Slider(-0.70, 0.70, value=float(initial[1]), step=0.005, label="Target y (m)")
                phi = gr.Slider(-180, 180, value=float(np.rad2deg(initial[2])), step=1, label="Target orientation φ (°)")
                button = gr.Button("Solve target", variant="primary")
                gr.Markdown(
                    "Some slider combinations are infeasible. A large residual means the selected "
                    "solver did not reproduce the pose; numerical failure alone does not establish infeasibility."
                )
            with gr.Column(scale=2):
                plot = gr.Plot(label="Arm pose comparison")
        results = gr.Markdown()
        button.click(solve, inputs=[x, y, phi], outputs=[plot, results])
        demo.load(solve, inputs=[x, y, phi], outputs=[plot, results])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_app(args.checkpoint).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
