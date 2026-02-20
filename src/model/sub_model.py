"""
PokéGrader — Sub-Models for Corner, Edge, and Surface Analysis

These models predict TAG-style sub-scores (0-1000 normalized to 0-1)
from individual corner/edge crop images. Unlike the overall grader
(classification into 10 grades), these are regression models that
predict continuous quality scores.

Architecture:
    EfficientNet-B0 backbone (pretrained)
    → Global average pooling
    → Dropout → FC → ReLU → Dropout → Multi-head output

Each model has multiple output heads:
    - Corner model: fray, fill, angle (3 heads)
    - Edge model: fray, fill, angle (3 heads)

Angle head uses a mask since not all corners/edges have angle scores.

Usage:
    from src.model.sub_models import CornerModel, EdgeModel

    corner_model = CornerModel(pretrained=True)
    preds = corner_model(images)  # Dict with 'fray', 'fill', 'angle' tensors

    edge_model = EdgeModel(pretrained=True)
    preds = edge_model(images)
"""

import torch
import torch.nn as nn
from torchvision import models


class SubScoreModel(nn.Module):
    """
    Base regression model for predicting TAG sub-scores from image crops.

    Shared architecture for corners and edges — both predict fray/fill/angle.
    Uses EfficientNet-B0 backbone with separate regression heads per score.

    Args:
        pretrained: Use ImageNet pretrained backbone
        dropout: Dropout rate
        hidden_dim: Hidden layer size before output heads
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension from EfficientNet-B0
        feat_dim = 1280

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # Separate output heads — each predicts a single score in [0, 1]
        self.head_fray = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head_fill = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head_angle = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            Dict with 'fray', 'fill', 'angle' — each (B, 1) in [0, 1]
        """
        features = self.features(x)
        pooled = self.pool(features).flatten(1)  # (B, 1280)
        shared = self.shared(pooled)              # (B, 256)

        return {
            "fray": self.head_fray(shared),    # (B, 1)
            "fill": self.head_fill(shared),    # (B, 1)
            "angle": self.head_angle(shared),  # (B, 1)
        }

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict scores scaled to TAG's 0-1000 range.

        Args:
            x: (B, 3, H, W) input images

        Returns:
            Dict with 'fray', 'fill', 'angle' — each (B,) in [0, 1000]
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)
            return {
                k: (v.squeeze(-1) * 1000).clamp(0, 1000)
                for k, v in raw.items()
            }

    def freeze_backbone(self):
        """Freeze EfficientNet backbone for Phase 1 training."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze for Phase 2 fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


# Aliases for clarity
class CornerModel(SubScoreModel):
    """EfficientNet model for corner quality scoring (fray/fill/angle)."""
    pass


class EdgeModel(SubScoreModel):
    """EfficientNet model for edge quality scoring (fray/fill/angle)."""
    pass


# ---------------------------------------------------------------------------
# Loss function for sub-score regression
# ---------------------------------------------------------------------------

class SubScoreLoss(nn.Module):
    """
    Multi-head regression loss for fray/fill/angle predictions.

    Handles missing angle scores by masking them out of the loss.
    Uses Smooth L1 (Huber) loss for robustness to outliers.

    The label tensor format is [fray, fill, angle] where angle = -1
    indicates a missing value that should be masked.

    Args:
        angle_weight: Relative weight for angle loss vs fray/fill.
                      Lower since angle is often missing and less consistent.
    """

    def __init__(self, angle_weight: float = 0.5):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.angle_weight = angle_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: Dict with 'fray', 'fill', 'angle' — each (B, 1)
            labels: (B, 3) tensor with [fray, fill, angle], angle=-1 if missing

        Returns:
            Scalar loss
        """
        fray_pred = predictions["fray"].squeeze(-1)   # (B,)
        fill_pred = predictions["fill"].squeeze(-1)    # (B,)
        angle_pred = predictions["angle"].squeeze(-1)  # (B,)

        fray_target = labels[:, 0]   # (B,)
        fill_target = labels[:, 1]   # (B,)
        angle_target = labels[:, 2]  # (B,), -1 for missing

        # Fray and fill loss (always present)
        fray_loss = self.loss_fn(fray_pred, fray_target).mean()
        fill_loss = self.loss_fn(fill_pred, fill_target).mean()

        # Angle loss (masked — only compute where angle >= 0)
        angle_mask = angle_target >= 0
        if angle_mask.any():
            angle_loss = self.loss_fn(
                angle_pred[angle_mask],
                angle_target[angle_mask],
            ).mean()
        else:
            angle_loss = torch.tensor(0.0, device=fray_pred.device)

        return fray_loss + fill_loss + self.angle_weight * angle_loss