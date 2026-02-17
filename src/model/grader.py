"""
PokéGrader — Card Condition Grading Model

Fine-tuned EfficientNet-B0 that predicts PSA grades (1-10) from card images.

Architecture:
    EfficientNet-B0 backbone (pretrained on ImageNet)
    → Global average pooling
    → Dropout
    → FC head → 10 classes (PSA 1-10)

The model predicts a grade distribution (softmax over 10 grades), which
gives us both a predicted grade and a confidence score. We use ordinal-aware
training so the model understands that predicting 8 when the answer is 9
is better than predicting 3.

Usage:
    from src.model.grader import CardGraderModel

    model = CardGraderModel()
    output = model(image_tensor)   # (batch, 10) logits
    grade = output.argmax(dim=1) + 1  # +1 because grades are 1-10, indices are 0-9
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CardGraderModel(nn.Module):
    """
    EfficientNet-B0 fine-tuned for PSA grade prediction.

    Input:  (B, 3, 224, 224) normalized card image
    Output: (B, 10) logits for PSA grades 1-10
    """

    def __init__(
        self,
        num_grades: int = 10,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Everything except the final classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # EfficientNet-B0 outputs 1280 features
        in_features = 1280

        # Grading head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_grades),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalized image tensor

        Returns:
            (B, 10) logits — use .argmax(dim=1) + 1 for grade prediction
        """
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)

    def predict_grade(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict grade with confidence.

        Returns:
            grades:      (B,) predicted PSA grades (1-10)
            confidences: (B,) confidence scores (0-1)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidences, indices = probs.max(dim=1)
        grades = indices + 1  # 0-indexed → 1-indexed
        return grades, confidences

    def freeze_backbone(self):
        """Freeze the EfficientNet backbone (train only the head)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


class OrdinalLoss(nn.Module):
    """
    Cross-entropy + ordinal penalty.

    Standard cross-entropy treats all wrong predictions equally.
    This adds a penalty scaled by how far off the prediction is,
    so predicting 8 when the answer is 9 is penalized less than
    predicting 3 when the answer is 9.

    Args:
        num_classes:    Number of grade classes (10).
        ordinal_weight: How much to weight the ordinal penalty (0-1).
    """

    def __init__(self, num_classes: int = 10, ordinal_weight: float = 0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight

        # Distance matrix: distance[i][j] = |i - j| / (num_classes - 1)
        distance = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                distance[i][j] = abs(i - j) / (num_classes - 1)
        self.register_buffer("distance", distance)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 10) raw model output
            targets: (B,) ground truth grades as indices (0-9)
        """
        ce_loss = self.ce(logits, targets)

        # Ordinal penalty: expected distance between prediction and target
        probs = torch.softmax(logits, dim=1)
        target_distances = self.distance[targets]
        ordinal_loss = (probs * target_distances).sum(dim=1).mean()

        return ce_loss + self.ordinal_weight * ordinal_loss