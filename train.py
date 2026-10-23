"""Train on CPU with Adam, validation early stopping, and a wall-time budget."""
import argparse
import csv
import json
import time
from pathlib import Path

from runtime import configure_plotting
configure_plotting()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import DEFAULT_DATA, ROOT, load_dataset, save_dataset
from model import DEFAULT_CHECKPOINT, IKModel, decode_angles, fk_torch, ik_loss, load_model


def _portable_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def plot_history(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    epochs = [r["epoch"] for r in rows]
    for key, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        axes[0].plot(epochs, [r[key] for r in rows], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Composite loss", yscale="log", title="Training and validation")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_position_mm"] for r in rows], color="#0d9488")
    axes[1].set(xlabel="Epoch", ylabel="Mean position error (mm)", title="Validation accuracy")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if not args.data.exists():
        save_dataset(args.data, samples=200_000, seed=args.seed)
    data = load_dataset(args.data)
    x_train = torch.from_numpy(data["pose_train"].astype(np.float32))
    q_train = torch.from_numpy(data["q_train"].astype(np.float32))
    x_val = torch.from_numpy(data["pose_val"].astype(np.float32))
    q_val = torch.from_numpy(data["q_val"].astype(np.float32))
    model = IKModel(x_train.mean(0), x_train.std(0, unbiased=False))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.checkpoint.parent / "history.csv"
    rows, best, stale, best_epoch = [], float("inf"), 0, 0
    early_best = float("inf")
    reason = "epoch limit"
    generator = torch.Generator().manual_seed(args.seed)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_position_mm", "elapsed_seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(x_train), generator=generator)
            total = 0.0
            seen = 0
            for indices in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = ik_loss(model(x_train[indices]), x_train[indices], q_train[indices])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
                seen += len(indices)
                if time.perf_counter() - started >= args.max_seconds:
                    break
            model.eval()
            val_total, position_total = 0.0, 0.0
            with torch.inference_mode():
                for x, q in zip(x_val.split(args.batch_size), q_val.split(args.batch_size)):
                    outputs = model(x)
                    val_total += ik_loss(outputs, x, q).item() * len(x)
                    error = torch.linalg.vector_norm(fk_torch(decode_angles(outputs))[..., :2] - x[..., :2], dim=-1)
                    position_total += error.sum().item()
            validation = val_total / len(x_val)
            row = dict(epoch=epoch, train_loss=total / seen, val_loss=validation,
                       val_position_mm=1000 * position_total / len(x_val),
                       elapsed_seconds=time.perf_counter() - started)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if validation < best:
                best, best_epoch = validation, epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "validation_loss": validation, "seed": args.seed}, args.checkpoint)
            if validation < early_best - args.min_delta:
                early_best, stale = validation, 0
            else:
                stale += 1
            scheduler.step(validation)
            if epoch == 1 or epoch % 5 == 0:
                print(f"epoch {epoch:3d} train={row['train_loss']:.6f} val={validation:.6f} "
                      f"position={row['val_position_mm']:.2f} mm elapsed={row['elapsed_seconds']:.1f}s", flush=True)
            if time.perf_counter() - started >= args.max_seconds:
                reason = "wall-time budget"
                break
            if stale >= args.patience:
                reason = "validation early stopping"
                break
    plot_history(rows, args.checkpoint.parent / "training_curve.png")
    metadata = {"seed": args.seed, "threads": args.threads, "epochs_completed": len(rows),
                "best_epoch": best_epoch, "best_validation_loss": best, "stop_reason": reason,
                "training_seconds": time.perf_counter() - started,
                "train_samples": len(x_train), "val_samples": len(x_val),
                "hyperparameters": {k: _portable_path(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()}}
    (args.checkpoint.parent / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return load_model(args.checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--max-seconds", type=float, default=360)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.patience) < 1 or args.max_seconds <= 0:
        parser.error("epochs, batch size, threads, patience and max seconds must be positive")
    train(args)


if __name__ == "__main__":
    main()
