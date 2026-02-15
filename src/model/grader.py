"""
PokéGrader — Card Condition Grading Model

Architecture: Fine-tuned EfficientNet with multi-head output
for grading centering, corners, edges, and surface quality.

TODO: Implement once training data is available.
"""

import torch
import torch.nn as nn


class CardGraderModel(nn.Module):
    """
    Multi-head CNN that outputs sub-grades for each grading factor.
    
    Inputs:
        - Card image (224x224 RGB)
    
    Outputs:
        - centering_score (float, 0-1)
        - corners_score (float, 0-1)  
        - edges_score (float, 0-1)
        - surface_score (float, 0-1)
    """

    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        # TODO: Load pretrained backbone
        # TODO: Add grading heads (centering, corners, edges, surface)
        raise NotImplementedError("Model will be implemented once training data is collected.")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError
