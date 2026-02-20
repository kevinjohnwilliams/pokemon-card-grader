"""
PokéGrader — TAG Data Training Pipeline

Trains corner, edge, and overall grading models using scraped TAG data.

This replaces the ImageFolder-based training approach with one that uses
TAG's rich sub-scores (fray/fill/angle per corner and edge, 0-1000).

Usage:
    # Train corner model
    python -m src.model.train_tag --model corners --data-dir tag_dataset/cards

    # Train edge model
    python -m src.model.train_tag --model edges --data-dir tag_dataset/cards

    # Train overall grade model
    python -m src.model.train_tag --model overall --data-dir tag_dataset/cards

    # Full pipeline — train all three
    python -m src.model.train_tag --model all --data-dir tag_dataset/cards
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.tag_dataset import (
    TAGCornerDataset,
    TAGEdgeDataset,
    TAGCardDataset,
    dataset_stats,
)
from src.model.sub_models import CornerModel, EdgeModel, SubScoreLoss
from src.model.grader import CardGraderModel, OrdinalLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(phase: str, crop_type: str = "corner"):
    """
    Build image transforms.

    Args:
        phase: 'train' or 'val'
        crop_type: 'corner', 'edge', or 'card' — determines input size
    """
    # Corner/edge crops are small, cards are larger
    img_size = 128 if crop_type in ("corner", "edge") else 224

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Training loop (shared by all model types)
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        if model_type == "overall":
            images, labels, _meta = batch
            images = images.to(device)
            targets = labels["grade_class"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
        else:
            images, labels, _meta = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, model_type):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    # For regression models: track per-head MAE
    head_errors = {"fray": [], "fill": [], "angle": []}
    # For overall model: track accuracy
    correct = 0
    within_1 = 0
    total = 0

    for batch in dataloader:
        if model_type == "overall":
            images, labels, _meta = batch
            images = images.to(device)
            targets = labels["grade_class"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            within_1 += ((preds - targets).abs() <= 1).sum().item()
            total += targets.size(0)
        else:
            images, labels, _meta = batch
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)

            # Per-head MAE (in TAG 0-1000 scale)
            for head in ("fray", "fill", "angle"):
                pred = predictions[head].squeeze(-1)
                target = labels[:, {"fray": 0, "fill": 1, "angle": 2}[head]]

                if head == "angle":
                    mask = target >= 0
                    if mask.any():
                        mae = (pred[mask] - target[mask]).abs().mean().item() * 1000
                        head_errors[head].append(mae)
                else:
                    mae = (pred - target).abs().mean().item() * 1000
                    head_errors[head].append(mae)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    metrics = {"loss": avg_loss}

    if model_type == "overall":
        metrics["accuracy"] = correct / max(total, 1)
        metrics["within_1"] = within_1 / max(total, 1)
    else:
        for head in ("fray", "fill", "angle"):
            if head_errors[head]:
                metrics[f"{head}_mae"] = np.mean(head_errors[head])

    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    model_type: str,
    data_dir: str,
    output_dir: str = "data/models",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_epochs: int = 5,
    patience: int = 7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 4,
):
    """
    Train a model on TAG data.

    Args:
        model_type: 'corners', 'edges', or 'overall'
        data_dir: Path to TAG cards directory
        output_dir: Where to save trained models
        epochs: Maximum training epochs
        batch_size: Batch size
        lr: Learning rate for Phase 2 (Phase 1 uses 10x)
        freeze_epochs: Number of Phase 1 epochs (frozen backbone)
        patience: Early stopping patience
        val_ratio: Fraction of cards to use for validation
        seed: Random seed
        num_workers: DataLoader workers
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_type} model on {device}")

    # ---- Create datasets ----
    if model_type == "corners":
        crop_type = "corner"
        full_dataset = TAGCornerDataset(
            data_dir,
            transform=None,  # Applied per-split below
        )
        train_ds, val_ds = full_dataset.split(val_ratio=val_ratio, seed=seed)
        train_ds.transform = get_transforms("train", crop_type)
        val_ds.transform = get_transforms("val", crop_type)

        model = CornerModel(pretrained=True).to(device)
        criterion = SubScoreLoss(angle_weight=0.5)

    elif model_type == "edges":
        crop_type = "edge"
        full_dataset = TAGEdgeDataset(
            data_dir,
            transform=None,
            skip_slab_images=True,
        )
        train_ds, val_ds = full_dataset.split(val_ratio=val_ratio, seed=seed)
        train_ds.transform = get_transforms("train", crop_type)
        val_ds.transform = get_transforms("val", crop_type)

        model = EdgeModel(pretrained=True).to(device)
        criterion = SubScoreLoss(angle_weight=0.5)

    elif model_type == "overall":
        crop_type = "card"
        full_dataset = TAGCardDataset(
            data_dir,
            transform=None,
        )
        train_ds, val_ds = full_dataset.split(val_ratio=val_ratio, seed=seed)
        train_ds.transform = get_transforms("train", crop_type)
        val_ds.transform = get_transforms("val", crop_type)

        model = CardGraderModel(pretrained=True).to(device)
        criterion = OrdinalLoss(num_classes=10, ordinal_weight=0.3)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ---- Print dataset stats ----
    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val:   {len(val_ds)} samples")
    stats = dataset_stats(train_ds)
    logger.info(f"Dataset stats: {json.dumps(stats, indent=2, default=str)}")

    # ---- Data loaders ----
    # Custom collate for overall model (dict labels)
    if model_type == "overall":
        def collate_fn(batch):
            images = torch.stack([b[0] for b in batch])
            labels = {
                "grade_class": torch.stack([b[1]["grade_class"] for b in batch]),
                "tag_score": torch.stack([b[1]["tag_score"] for b in batch]),
            }
            meta = [b[2] for b in batch]
            return images, labels, meta
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---- Setup output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{model_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Frozen backbone ----
    logger.info(f"=== Phase 1: Frozen backbone ({freeze_epochs} epochs, lr={lr * 10:.1e}) ===")
    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = run_dir / "best_model.pt"

    for epoch in range(freeze_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        elapsed = time.time() - t0

        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        logger.info(f"  Epoch {epoch+1}/{freeze_epochs} [{elapsed:.1f}s] — "
                     f"train_loss: {train_loss:.4f} | {metric_str}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(model, optimizer, epoch, val_metrics, model_type, best_model_path)

    # ---- Phase 2: Full fine-tuning ----
    remaining_epochs = epochs - freeze_epochs
    logger.info(f"=== Phase 2: Full fine-tuning ({remaining_epochs} epochs, lr={lr:.1e}) ===")
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )

    for epoch in range(remaining_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        elapsed = time.time() - t0

        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        logger.info(f"  Epoch {freeze_epochs + epoch + 1}/{epochs} [{elapsed:.1f}s] — "
                     f"train_loss: {train_loss:.4f} | {metric_str}")

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            _save_checkpoint(model, optimizer, freeze_epochs + epoch, val_metrics, model_type, best_model_path)
            logger.info(f"  ✅ New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"  ⏹ Early stopping after {patience} epochs without improvement")
            break

    # ---- Save training summary ----
    summary = {
        "model_type": model_type,
        "data_dir": data_dir,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_loss": best_val_loss,
        "epochs_completed": freeze_epochs + epoch + 1,
        "timestamp": timestamp,
        "device": str(device),
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training complete. Best model: {best_model_path}")
    return str(best_model_path)


def _save_checkpoint(model, optimizer, epoch, metrics, model_type, path):
    """Save model checkpoint."""
    torch.save({
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PokéGrader models on TAG data")
    parser.add_argument("--model", choices=["corners", "edges", "overall", "all"],
                        required=True, help="Which model to train")
    parser.add_argument("--data-dir", required=True,
                        help="Path to TAG cards directory")
    parser.add_argument("--output-dir", default="data/models",
                        help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    if args.model == "all":
        for m in ["corners", "edges", "overall"]:
            logger.info(f"\n{'='*60}\nTraining {m} model\n{'='*60}")
            train_model(
                model_type=m,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                freeze_epochs=args.freeze_epochs,
                patience=args.patience,
                val_ratio=args.val_ratio,
                seed=args.seed,
                num_workers=args.workers,
            )
    else:
        train_model(
            model_type=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            patience=args.patience,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()