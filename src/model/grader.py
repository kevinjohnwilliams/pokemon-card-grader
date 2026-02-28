"""
PokéGrader — Card Grading Model (v2 — Regression with Auxiliary Features)

Single EfficientNet-B0 model that predicts a continuous grade (1.0-10.0)
from the full card image, augmented with structured metadata features
(defect counts, centering offsets, defect type flags).

Architecture:
    Image → EfficientNet-B0 → 1280-dim features
    Aux features (11-dim) → small MLP → 32-dim
    Concatenate → [1312-dim] → FC layers → single output (1.0-10.0)

The model learns visual condition from pixels AND leverages the structured
defect/centering data that's already in the metadata. This hybrid approach
outperforms either signal alone.

Usage:
    from src.model.grader import CardGrader

    model = CardGrader(pretrained=True)
    grade = model.predict(image_tensor, aux_tensor)  # Returns e.g. 8.7
"""

import torch
import torch.nn as nn
from torchvision import models


class CardGrader(nn.Module):
    """
    Hybrid image + metadata model for card grade prediction.

    Combines EfficientNet-B0 visual features with structured auxiliary
    features (defect counts, centering offsets) to predict a continuous
    grade from 1.0 to 10.0.

    Args:
        num_aux_features: Number of auxiliary metadata features (default: 11)
        pretrained: Use ImageNet pretrained backbone
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        num_aux_features: int = 11,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ---- Image branch: EfficientNet-B0 ----
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        img_dim = 1280  # EfficientNet-B0 output dimension

        # ---- Aux branch: small MLP for structured features ----
        aux_hidden = 32
        self.aux_net = nn.Sequential(
            nn.Linear(num_aux_features, aux_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # ---- Combined head ----
        combined_dim = img_dim + aux_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Store config for checkpoint saving
        self.config = {
            "num_aux_features": num_aux_features,
            "pretrained": pretrained,
            "dropout": dropout,
        }

    def forward(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) card images
            aux:    (B, 11) auxiliary features

        Returns:
            (B, 1) predicted grades (raw — not clamped)
        """
        # Image features
        img_features = self.features(images)
        img_features = self.pool(img_features).flatten(1)  # (B, 1280)

        # Aux features
        aux_features = self.aux_net(aux)  # (B, 32)

        # Combine and predict
        combined = torch.cat([img_features, aux_features], dim=1)  # (B, 1312)
        return self.head(combined)  # (B, 1)

    def predict(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict grades clamped to valid 1.0-10.0 range.

        Args:
            images: (B, 3, 224, 224) card images
            aux:    (B, 11) auxiliary features

        Returns:
            (B,) predicted grades in [1.0, 10.0]
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(images, aux).squeeze(-1)
            return raw.clamp(1.0, 10.0)

    def predict_with_confidence(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
    ) -> dict:
        """
        Predict grade with a simple confidence estimate.

        Confidence is based on how close the prediction is to a whole
        or half grade (predictions near X.0 or X.5 are more confident
        than those near X.25 or X.75, which suggest the model is
        between two grades).

        Returns:
            Dict with 'grade', 'rounded_grade', 'confidence'
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(images, aux).squeeze(-1).clamp(1.0, 10.0)

            # Round to nearest 0.5
            rounded = (raw * 2).round() / 2

            # Confidence: how close prediction is to the rounded value
            # Max confidence when prediction == rounded (distance = 0)
            # Min confidence when prediction is 0.25 away (between grades)
            distance = (raw - rounded).abs()
            confidence = 1.0 - (distance / 0.25).clamp(0, 1)

            return {
                "grade": raw,
                "rounded_grade": rounded,
                "confidence": confidence,
            }

    def freeze_backbone(self):
        """Freeze EfficientNet backbone for Phase 1 training."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze for Phase 2 fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class GradeLoss(nn.Module):
    """
    Combined regression loss for grade prediction.

    Smooth L1 (Huber) loss for robustness to outlier labels, plus an
    optional ordinal penalty that adds extra cost when predictions are
    far from the target (e.g., predicting 3 for a grade 9 card).

    Args:
        ordinal_weight: Weight for the ordinal distance penalty.
        huber_delta: Delta for Smooth L1 loss (transition point
                     between L1 and L2 behavior).
    """

    def __init__(self, ordinal_weight: float = 0.1, huber_delta: float = 1.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_delta)
        self.ordinal_weight = ordinal_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, 1) or (B,) predicted grades
            targets:     (B,) target grades (1.0 - 10.0)
        """
        preds = predictions.squeeze(-1)

        # Primary loss: Smooth L1
        huber_loss = self.huber(preds, targets)

        # Ordinal penalty: squared distance normalized by grade range
        # This adds extra penalty for being way off
        if self.ordinal_weight > 0:
            distance = (preds - targets).abs()
            ordinal_penalty = (distance / 9.0).pow(2).mean()  # Normalized by max range
            return huber_loss + self.ordinal_weight * ordinal_penalty

        return huber_loss