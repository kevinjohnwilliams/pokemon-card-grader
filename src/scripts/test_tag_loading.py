"""
Quick smoke test for TAG dataset loading.

Run from project root:
    python scripts/test_tag_loading.py tag_dataset/cards
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tag_dataset import (
    TAGCornerDataset,
    TAGEdgeDataset,
    TAGCardDataset,
    dataset_stats,
    load_all_metadata,
)
from src.model.sub_models import CornerModel, EdgeModel, SubScoreLoss
from torchvision import transforms


def main(cards_dir: str):
    print("=" * 60)
    print("TAG DATASET SMOKE TEST")
    print("=" * 60)

    # ---- Test metadata loading ----
    print("\n1. Loading metadata...")
    metadata = load_all_metadata(cards_dir)
    print(f"   ✅ Loaded {len(metadata)} cards")

    # ---- Test corner dataset ----
    print("\n2. Creating corner dataset...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    corner_ds = TAGCornerDataset(cards_dir, transform=transform)
    print(f"   ✅ {len(corner_ds)} corner samples")

    if len(corner_ds) > 0:
        img, label, meta = corner_ds[0]
        print(f"   Sample: {meta['cert_id']} {meta['side']} {meta['position']}")
        print(f"   Image shape: {img.shape}")
        print(f"   Labels: fray={label[0]:.3f} fill={label[1]:.3f} angle={label[2]:.3f}")

    # ---- Test edge dataset ----
    print("\n3. Creating edge dataset...")
    edge_ds = TAGEdgeDataset(cards_dir, transform=transform, skip_slab_images=True)
    print(f"   ✅ {len(edge_ds)} edge samples (slab images filtered)")

    # ---- Test card dataset ----
    print("\n4. Creating card dataset...")
    card_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    card_ds = TAGCardDataset(cards_dir, transform=card_transform)
    print(f"   ✅ {len(card_ds)} card samples")

    if len(card_ds) > 0:
        img, label, meta = card_ds[0]
        print(f"   Sample: {meta['cert_id']} grade={meta['grade']} tag_score={meta['tag_score']}")

    # ---- Test splitting ----
    print("\n5. Testing train/val split...")
    train_ds, val_ds = corner_ds.split(val_ratio=0.15, seed=42)
    print(f"   ✅ Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Verify no card leakage
    train_cards = set(s["cert_id"] for s in train_ds.samples)
    val_cards = set(s["cert_id"] for s in val_ds.samples)
    overlap = train_cards & val_cards
    if overlap:
        print(f"   ❌ LEAKAGE: {len(overlap)} cards in both splits!")
    else:
        print(f"   ✅ No card leakage between splits")

    # ---- Test model forward pass ----
    print("\n6. Testing model forward pass...")
    model = CornerModel(pretrained=False)  # No pretrained to save download time
    dummy = torch.randn(2, 3, 128, 128)
    preds = model(dummy)
    print(f"   ✅ Corner model output: fray={preds['fray'].shape}, "
          f"fill={preds['fill'].shape}, angle={preds['angle'].shape}")

    # ---- Test loss ----
    print("\n7. Testing SubScoreLoss...")
    criterion = SubScoreLoss(angle_weight=0.5)
    labels = torch.tensor([[0.95, 0.98, 0.99], [0.80, 0.85, -1.0]])
    loss = criterion(preds, labels)
    print(f"   ✅ Loss: {loss.item():.4f}")

    # ---- Dataset stats ----
    print("\n8. Dataset statistics...")
    stats = dataset_stats(corner_ds)
    print(f"   Unique cards: {stats['unique_cards']}")
    print(f"   Fray range: {stats['fray']['min']:.3f} - {stats['fray']['max']:.3f}")
    print(f"   Fill range: {stats['fill']['min']:.3f} - {stats['fill']['max']:.3f}")
    print(f"   Has angle: {stats['has_angle']} | Missing: {stats['missing_angle']}")
    print(f"   By side: {stats['by_side']}")

    card_stats = dataset_stats(card_ds)
    if "grade_distribution" in card_stats:
        print(f"   Grade distribution: {card_stats['grade_distribution']}")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_tag_loading.py <cards_dir>")
        sys.exit(1)
    main(sys.argv[1])