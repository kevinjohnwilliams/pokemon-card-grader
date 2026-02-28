"""
PokéGrader — Training Pipeline (v2 — Single Regression Model)

Trains the CardGrader model to predict continuous grades (1.0-10.0)
from full card images + auxiliary metadata features.

Two-phase training:
    Phase 1: Freeze EfficientNet backbone, train head + aux branch only
    Phase 2: Unfreeze everything, fine-tune at lower learning rate

Usage:
    python -m src.model.train_tag --data-dir data/raw
    python -m src.model.train_tag --data-dir data/raw --epochs 40 --batch-size 16
    python -m src.model.train_tag --data-dir data/raw --lr 0.0005 --freeze-epochs 8
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.tag_dataset import TAGCardDataset, dataset_stats
from src.model.grader import CardGrader, GradeLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(phase: str):
    """Build image transforms for train/val."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.02,
            ),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Custom collate that handles the (image, label, aux, meta) tuple."""
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    aux = torch.stack([b[2] for b in batch])
    meta = [b[3] for b in batch]
    return images, labels, aux, meta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for images, labels, aux, _meta in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        aux = aux.to(device)

        optimizer.zero_grad()
        predictions = model(images, aux)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.

    Returns dict with:
        loss:       Average loss
        mae:        Mean absolute error (in grade units)
        within_0_5: Fraction of predictions within 0.5 grades
        within_1:   Fraction of predictions within 1.0 grades
        exact:      Fraction matching when both rounded to nearest int
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    all_preds = []
    all_targets = []

    for images, labels, aux, _meta in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        aux = aux.to(device)

        predictions = model(images, aux)
        loss = criterion(predictions, labels)

        preds = predictions.squeeze(-1).clamp(1.0, 10.0)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

        total_loss += loss.item()
        n_batches += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    errors = (all_preds - all_targets).abs()

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "mae": errors.mean().item(),
        "within_0_5": (errors <= 0.5).float().mean().item(),
        "within_1": (errors <= 1.0).float().mean().item(),
        "exact": (all_preds.round() == all_targets.round()).float().mean().item(),
    }

    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
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
    ordinal_weight: float = 0.1,
):
    """
    Train the CardGrader model.

    Args:
        data_dir:        Path to card folders (data/raw)
        output_dir:      Where to save models
        epochs:          Total epochs (Phase 1 + Phase 2)
        batch_size:      Batch size
        lr:              Phase 2 learning rate (Phase 1 uses 10x)
        freeze_epochs:   Phase 1 epochs (frozen backbone)
        patience:        Early stopping patience
        val_ratio:       Validation split ratio
        seed:            Random seed
        num_workers:     DataLoader workers
        ordinal_weight:  Weight for ordinal distance penalty in loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Load dataset ----
    logger.info(f"Loading data from {data_dir}...")
    full_dataset = TAGCardDataset(data_dir)

    if len(full_dataset) == 0:
        logger.error("No valid samples found!")
        return

    train_ds, val_ds = full_dataset.split(val_ratio=val_ratio, seed=seed)
    train_ds.transform = get_transforms("train")
    val_ds.transform = get_transforms("val")

    # ---- Print stats ----
    stats = dataset_stats(full_dataset)
    logger.info(f"Total cards: {stats['total_samples']}")
    logger.info(f"Label sources: {stats['label_sources']}")
    logger.info(f"Data formats: {stats['data_formats']}")
    logger.info(f"Grade distribution: {stats['grade_distribution']}")
    logger.info(f"Half grades: {stats['half_grades']}")
    logger.info(f"TAG scores: {stats['tag_scores']['count']} cards")
    logger.info(f"Grade range: {stats['continuous_grade_range']}")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Data loaders ----
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

    # ---- Model and loss ----
    model = CardGrader(pretrained=True).to(device)
    criterion = GradeLoss(ordinal_weight=ordinal_weight)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "best_model.pt"
    logger.info(f"Output: {run_dir}")

    # ---- Phase 1: Frozen backbone ----
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 1: Frozen backbone ({freeze_epochs} epochs, lr={lr * 10:.1e})")
    logger.info(f"{'='*60}")

    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,} / {total_params:,}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10,
    )

    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(freeze_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            f"  Epoch {epoch+1}/{freeze_epochs} [{elapsed:.0f}s] — "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"MAE: {val_metrics['mae']:.3f} | "
            f"±0.5: {val_metrics['within_0_5']:.1%} | "
            f"±1.0: {val_metrics['within_1']:.1%} | "
            f"exact: {val_metrics['exact']:.1%}"
        )

        history.append({"epoch": epoch + 1, "phase": 1, "train_loss": train_loss, **val_metrics})

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            _save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)

    # ---- Phase 2: Full fine-tuning ----
    remaining = epochs - freeze_epochs
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2: Full fine-tuning ({remaining} epochs, lr={lr:.1e})")
    logger.info(f"{'='*60}")

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,} / {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )

    for epoch in range(remaining):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        global_epoch = freeze_epochs + epoch + 1
        logger.info(
            f"  Epoch {global_epoch}/{epochs} [{elapsed:.0f}s] — "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"MAE: {val_metrics['mae']:.3f} | "
            f"±0.5: {val_metrics['within_0_5']:.1%} | "
            f"±1.0: {val_metrics['within_1']:.1%} | "
            f"exact: {val_metrics['exact']:.1%}"
        )

        history.append({"epoch": global_epoch, "phase": 2, "train_loss": train_loss, **val_metrics})

        scheduler.step(val_metrics["mae"])

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            _save_checkpoint(model, optimizer, global_epoch, val_metrics, best_model_path)
            logger.info(f"  ✅ New best (MAE: {best_mae:.3f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(f"  ⏹ Early stopping after {patience} epochs without improvement")
            break

    # ---- Save training summary ----
    summary = {
        "timestamp": timestamp,
        "data_dir": data_dir,
        "total_cards": len(full_dataset),
        "train_cards": len(train_ds),
        "val_cards": len(val_ds),
        "best_mae": best_mae,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "device": str(device),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "freeze_epochs": freeze_epochs,
            "patience": patience,
            "ordinal_weight": ordinal_weight,
        },
        "history": history,
    }

    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best MAE: {best_mae:.3f} grades")
    logger.info(f"Model saved: {best_model_path}")
    logger.info(f"Summary: {run_dir / 'training_summary.json'}")

    return str(best_model_path)


def _save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint with all info needed for loading."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PokéGrader model")
    parser.add_argument("--data-dir", required=True,
                        help="Path to card data directory (e.g. data/raw)")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ordinal-weight", type=float, default=0.1)

    args = parser.parse_args()

    train(
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
        ordinal_weight=args.ordinal_weight,
    )


if __name__ == "__main__":
    main()