"""
PokéGrader — Training Pipeline

Trains the EfficientNet card grading model.

Dataset structure expected (ImageFolder format):
    data/raw/                  ← Scraped eBay images (TRAINING)
        grade_01/*.jpg
        grade_02/*.jpg
        ...
        grade_10/*.jpg
    data/processed/val/        ← Our PSA phone photos (VALIDATION)
        grade_01/*.jpg ... grade_10/*.jpg
    data/processed/test/       ← Held-out PSA phone photos (TEST)
        grade_01/*.jpg ... grade_10/*.jpg

Training strategy:
    Phase 1 — Freeze backbone, train head only (fast convergence)
    Phase 2 — Unfreeze backbone, fine-tune everything at lower LR

Usage:
    python -m src.model.train
    python -m src.model.train --config configs/default.yaml
    python -m src.model.train --train-dir data/augmented --val-dir data/processed/val
    python -m src.model.train --epochs 30 --batch-size 16 --lr 0.0005
"""

import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.model.grader import CardGraderModel, OrdinalLoss
from src.utils.augmentation import PhoneCameraTransform


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------

def build_transforms(phase: str, augment_intensity: str = "medium"):
    """
    Build image transforms for each phase.

    Train: phone camera augmentation + normalize
    Val/Test: just resize + normalize (clean evaluation)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            PhoneCameraTransform(intensity=augment_intensity),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def load_datasets(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    augment_intensity: str = "medium",
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Load datasets from directory structure.

    Expects ImageFolder layout: dir/grade_XX/*.jpg
    Returns dict of DataLoaders.
    """
    loaders = {}

    # Training data (scraped images, possibly augmented on disk)
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=build_transforms("train", augment_intensity),
    )
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation data (our PSA phone photos)
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=build_transforms("val"),
    )
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test data (held-out PSA phone photos)
    if test_dir and Path(test_dir).exists():
        test_dataset = datasets.ImageFolder(
            test_dir,
            transform=build_transforms("test"),
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    print(f"  Train: {len(train_dataset)} images, {len(train_dataset.classes)} grades")
    print(f"  Val:   {len(val_dataset)} images")
    if "test" in loaders:
        print(f"  Test:  {len(loaders['test'].dataset)} images")
    print(f"  Classes: {train_dataset.class_to_idx}")

    return loaders


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_dir: str = "data/raw",
    val_dir: str = "data/processed/val",
    test_dir: Optional[str] = "data/processed/test",
    output_dir: str = "data/models",
    config_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    lr_finetune: float = 1e-4,
    freeze_epochs: int = 5,
    patience: int = 7,
    dropout: float = 0.3,
    ordinal_weight: float = 0.3,
    augment_intensity: str = "medium",
    num_workers: int = 4,
):
    """
    Full two-phase training pipeline.

    Phase 1: Freeze backbone, train classification head only.
    Phase 2: Unfreeze, fine-tune everything at lower learning rate.

    Saves best model based on validation accuracy.
    """

    # Load config if provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        tc = cfg.get("training", {})
        epochs = tc.get("epochs", epochs)
        batch_size = tc.get("batch_size", batch_size)
        lr = tc.get("learning_rate", lr)
        patience = tc.get("early_stopping_patience", patience)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # Load data
    print("\n  Loading datasets...")
    loaders = load_datasets(
        train_dir, val_dir, test_dir,
        augment_intensity=augment_intensity,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Model
    print("\n  Initializing EfficientNet-B0...")
    model = CardGraderModel(dropout=dropout, pretrained=True).to(device)
    criterion = OrdinalLoss(num_classes=10, ordinal_weight=ordinal_weight).to(device)

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tracking
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_within1": [], "val_within1": [],
        "lr": [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"  Training PokéGrader — {epochs} epochs max")
    print(f"  Phase 1: Frozen backbone ({freeze_epochs} epochs, lr={lr})")
    print(f"  Phase 2: Full fine-tune (lr={lr_finetune})")
    print(f"  Early stopping: {patience} epochs patience")
    print(f"{'='*60}\n")

    for epoch in range(epochs):

        # --- Phase transitions ---
        if epoch == 0:
            model.freeze_backbone()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3,
            )
            print("  ❄️  Phase 1: Backbone frozen\n")

        elif epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=lr_finetune)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3,
            )
            print("\n  🔥 Phase 2: Full fine-tuning\n")

        # --- Train ---
        tm = _run_epoch(model, loaders["train"], criterion, device,
                        optimizer=optimizer, is_training=True)

        # --- Validate ---
        vm = _run_epoch(model, loaders["val"], criterion, device,
                        optimizer=None, is_training=False)

        # Record
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tm["loss"])
        history["val_loss"].append(vm["loss"])
        history["train_acc"].append(tm["accuracy"])
        history["val_acc"].append(vm["accuracy"])
        history["train_within1"].append(tm["within_1"])
        history["val_within1"].append(vm["within_1"])
        history["lr"].append(current_lr)

        scheduler.step(vm["accuracy"])

        # Print
        print(
            f"  Epoch {epoch+1:3d}/{epochs}  │  "
            f"Train: loss={tm['loss']:.4f} acc={tm['accuracy']:.1f}% ±1={tm['within_1']:.1f}%  │  "
            f"Val: loss={vm['loss']:.4f} acc={vm['accuracy']:.1f}% ±1={vm['within_1']:.1f}%  │  "
            f"lr={current_lr:.6f}"
        )

        # Best model
        if vm["accuracy"] > best_val_acc:
            best_val_acc = vm["accuracy"]
            best_epoch = epoch + 1
            no_improve = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "val_within_1": vm["within_1"],
            }, run_dir / "best_model.pt")

            print(f"           └── 💾 New best (val acc: {best_val_acc:.1f}%)")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch+1}")
            break

    # --- Test evaluation ---
    print(f"\n{'='*60}")
    print(f"  Best model: epoch {best_epoch}, val acc {best_val_acc:.1f}%")

    if "test" in loaders:
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test = _run_epoch(model, loaders["test"], criterion, device,
                          optimizer=None, is_training=False)

        print(f"\n  📊 Test Results (held-out PSA phone photos):")
        print(f"     Exact accuracy:  {test['accuracy']:.1f}%")
        print(f"     Within ±1 grade: {test['within_1']:.1f}%")
        print(f"     Avg grade error: {test['avg_distance']:.2f}")

        history["test_accuracy"] = test["accuracy"]
        history["test_within_1"] = test["within_1"]
        history["test_avg_error"] = test["avg_distance"]

    # Save artifacts
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    run_config = {
        "train_dir": train_dir, "val_dir": val_dir, "test_dir": test_dir,
        "epochs_run": epoch + 1, "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "batch_size": batch_size, "lr": lr, "lr_finetune": lr_finetune,
        "freeze_epochs": freeze_epochs, "dropout": dropout,
        "ordinal_weight": ordinal_weight, "augment_intensity": augment_intensity,
        "device": str(device), "timestamp": timestamp,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n  Saved → {run_dir}")
    print(f"{'='*60}\n")

    return run_dir


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, device, optimizer=None, is_training=True):
    """Run one epoch of training or evaluation."""

    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    within_1 = 0
    total = 0
    total_distance = 0.0

    ctx = torch.enable_grad() if is_training else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits.argmax(dim=1)
            n = labels.size(0)

            total_loss += loss.item() * n
            correct += (preds == labels).sum().item()
            within_1 += ((preds - labels).abs() <= 1).sum().item()
            total_distance += (preds - labels).abs().float().sum().item()
            total += n

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
        "within_1": 100.0 * within_1 / max(total, 1),
        "avg_distance": total_distance / max(total, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PokéGrader model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default="data/raw")
    parser.add_argument("--val-dir", type=str, default="data/processed/val")
    parser.add_argument("--test-dir", type=str, default="data/processed/test")
    parser.add_argument("--output-dir", type=str, default="data/models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--augment", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        augment_intensity=args.augment,
        num_workers=args.workers,
    )