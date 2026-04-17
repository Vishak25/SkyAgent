#!/usr/bin/env python3
"""
Training script for the SkyCast ST-GNN flight delay prediction model.

Usage (from backend/):
  python -m src.models.delay_gnn.train --data data/raw/iem_metar_30d.csv --epochs 200

What it does:
  1. Loads IEM METAR CSV → builds hourly graph snapshots (dataset.py)
  2. Splits temporally: 70% train / 15% val / 15% test (no future leakage)
  3. Trains the STGNN (2-hop GCN, 5-dim input, 32 hidden) with MSE loss
  4. Early-stops on validation MAE, saves best weights
  5. Evaluates on held-out test set (MAE, RMSE, R², delay-bracket accuracy)

Outputs:
  - model_weights/stgnn_best.pt   (best validation checkpoint)
  - stdout: training curves + final evaluation report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Resolve imports relative to backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.delay_gnn.model import STGNN
from src.models.delay_gnn.dataset import METARGraphDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression + classification metrics."""
    # Regression
    residuals = preds - targets
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # R²
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    # Delay-bracket accuracy (±15 min)
    bracket_15 = float(np.mean(np.abs(residuals) <= 15.0) * 100)
    # Delay-bracket accuracy (±30 min)
    bracket_30 = float(np.mean(np.abs(residuals) <= 30.0) * 100)

    # Directional accuracy: does the model correctly flag delay > 15 min?
    pred_delayed = preds > 15.0
    true_delayed = targets > 15.0
    if true_delayed.sum() > 0:
        recall = float(np.logical_and(pred_delayed, true_delayed).sum() / true_delayed.sum() * 100)
    else:
        recall = 100.0
    if pred_delayed.sum() > 0:
        precision = float(np.logical_and(pred_delayed, true_delayed).sum() / pred_delayed.sum() * 100)
    else:
        precision = 100.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_15min_pct": bracket_15,
        "within_30min_pct": bracket_30,
        "delay_recall_pct": recall,
        "delay_precision_pct": precision,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Log-transform helpers (combat skewed targets)
# ---------------------------------------------------------------------------

def log1p_transform(y: torch.Tensor) -> torch.Tensor:
    """log(1 + y) — compresses large delays, stabilises training."""
    return torch.log1p(y)


def inv_log1p(y_log: torch.Tensor) -> torch.Tensor:
    """Inverse: exp(y) - 1."""
    return torch.expm1(y_log)


# ---------------------------------------------------------------------------
# Weighted Huber loss (handles skew + outliers)
# ---------------------------------------------------------------------------

def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss with sample weights that upweight delayed observations.
    Prevents the model from collapsing to predict the mean.
    """
    # Weight: sqrt(1 + target) — higher-delay samples get more gradient
    weights = torch.sqrt(1.0 + target)
    residual = pred - target
    abs_res = residual.abs()
    quadratic = torch.clamp(abs_res, max=delta)
    linear = abs_res - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return (loss * weights).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: STGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one epoch on log-transformed targets, return (avg_loss, avg_mae)."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index).squeeze(-1)  # [total_nodes]
        target_log = log1p_transform(batch.y)  # Train in log-space

        loss = weighted_huber_loss(out, target_log, delta=1.0)
        loss.backward()

        # Gradient clipping to stabilise training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # MAE in original minutes space for interpretability
        pred_mins = inv_log1p(out.detach()).clamp(min=0.0)
        total_loss += loss.item()
        total_mae += float((pred_mins - batch.y).abs().mean().item())
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: STGNN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate — returns metrics in original minutes space."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index).squeeze(-1)
        target_log = log1p_transform(batch.y)

        loss = weighted_huber_loss(out, target_log, delta=1.0)
        total_loss += loss.item()

        pred_mins = inv_log1p(out).clamp(min=0.0)
        total_mae += float((pred_mins - batch.y).abs().mean().item())
        n_batches += 1

        all_preds.append(pred_mins.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    avg_mae = total_mae / max(n_batches, 1)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return avg_loss, avg_mae, preds, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train ST-GNN on IEM METAR data")
    parser.add_argument("--data", type=str, default="data/raw/iem_metar_30d.csv",
                        help="Path to IEM METAR CSV")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (number of graph snapshots per batch)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (epochs without val MAE improvement)")
    parser.add_argument("--hidden", type=int, default=32,
                        help="Hidden channels in ST-GNN")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="model_weights",
                        help="Directory to save best model weights")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', 'mps', or 'auto'")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Loading METAR dataset...")
    print(f"{'='*60}")

    ds = METARGraphDataset(args.data, seed=args.seed)
    stats = ds.stats()

    print(f"  Total graph snapshots : {stats['total_snapshots']}")
    print(f"  Airports ({stats['num_airports']}): {', '.join(stats['airports'])}")
    print(f"  Edges per graph      : {stats['num_edges']}")
    print(f"  Delay mean           : {stats['delay_mean']:.1f} min")
    print(f"  Delay std            : {stats['delay_std']:.1f} min")
    print(f"  Delay median         : {stats['delay_median']:.1f} min")
    print(f"  Delay max            : {stats['delay_max']:.1f} min")
    print(f"  Non-zero delay %     : {stats['pct_nonzero_delay']:.1f}%")

    train_data, val_data, test_data = ds.split()
    print(f"\n  Train: {len(train_data)}  |  Val: {len(val_data)}  |  Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # 2. Initialise model
    # -----------------------------------------------------------------------
    model = STGNN(in_channels=5, hidden_channels=args.hidden, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-5
    )

    # -----------------------------------------------------------------------
    # 3. Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train MAE':>9} | {'Val Loss':>9} | {'Val MAE':>8} | {'LR':>8}")
    print("-" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "stgnn_best.pt"

    best_val_mae = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    t0 = time.time()

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, device)
        scheduler.step(epoch)

        lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
            "val_loss": val_loss, "val_mae": val_mae, "lr": lr,
        })

        improved = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            improved = " *"
        else:
            epochs_no_improve += 1

        if epoch <= 10 or epoch % 10 == 0 or improved or epochs_no_improve >= args.patience:
            print(f"{epoch:6d} | {train_loss:10.4f} | {train_mae:9.2f} | {val_loss:9.4f} | {val_mae:8.2f} | {lr:8.6f}{improved}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best val MAE: {best_val_mae:.2f} min (epoch {best_epoch})")

    # -----------------------------------------------------------------------
    # 4. Test evaluation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")

    # Load best weights
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_mae, test_preds, test_targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(test_preds, test_targets)

    print(f"\n  Test MAE             : {metrics['mae']:.2f} min")
    print(f"  Test RMSE            : {metrics['rmse']:.2f} min")
    print(f"  Test R²              : {metrics['r2']:.4f}")
    print(f"  Within ±15 min       : {metrics['within_15min_pct']:.1f}%")
    print(f"  Within ±30 min       : {metrics['within_30min_pct']:.1f}%")
    print(f"  Delay (>15m) recall  : {metrics['delay_recall_pct']:.1f}%")
    print(f"  Delay (>15m) precision: {metrics['delay_precision_pct']:.1f}%")

    # -----------------------------------------------------------------------
    # 5. Save training report
    # -----------------------------------------------------------------------
    report = {
        "dataset": str(args.data),
        "dataset_stats": stats,
        "model": {
            "in_channels": 5, "hidden_channels": args.hidden, "out_channels": 1,
            "parameters": n_params,
        },
        "training": {
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "elapsed_seconds": elapsed,
            "lr": args.lr, "batch_size": args.batch_size,
            "patience": args.patience, "seed": args.seed,
        },
        "test_metrics": metrics,
    }
    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Weights saved to : {best_path}")
    print(f"  Report saved to  : {report_path}")

    # Quick sample predictions
    print(f"\n{'='*60}")
    print("Sample predictions (test set, first snapshot):")
    print(f"{'='*60}")
    sample = test_data[0].to(device)
    model.eval()
    with torch.no_grad():
        sample_out_log = model(sample.x, sample.edge_index).squeeze(-1)
        sample_out = inv_log1p(sample_out_log).clamp(min=0.0)

    airports = ds._airports
    print(f"  {'Airport':>6} | {'Predicted':>9} | {'Actual':>9} | {'Error':>7}")
    print("  " + "-" * 42)
    for i, ap in enumerate(airports):
        pred = float(sample_out[i].item())
        actual = float(sample.y[i].item())
        err = pred - actual
        print(f"  {ap:>6} | {pred:9.1f} | {actual:9.1f} | {err:+7.1f}")

    print(f"\nDone. Model ready for inference at: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
