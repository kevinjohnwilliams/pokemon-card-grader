"""
PokéGrader — Image Preprocessing Pipeline

Combines card detection, cropping, and normalization into a
single pipeline for both analysis and future model inference.
"""

import cv2
import numpy as np
from typing import Optional

from src.utils.card_detector import detect_and_crop_card, ensure_portrait


def preprocess_card_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Full preprocessing pipeline: detect → crop → normalize.

    Args:
        image: Raw camera image (BGR)
        target_size: Output size for model input

    Returns:
        Tuple of:
            - model_input: Normalized image ready for model (or None)
            - card_image: Cropped card at full resolution (or None)
            - info: Detection metadata
    """
    # Step 1: Detect and crop
    cropped, _, info = detect_and_crop_card(image)

    if cropped is None:
        return None, None, info

    # Step 2: Ensure portrait
    card_image = ensure_portrait(cropped)

    # Step 3: Resize for model
    resized = cv2.resize(card_image, target_size, interpolation=cv2.INTER_AREA)

    # Step 4: Normalize (ImageNet stats, standard for pretrained models)
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std

    # Convert HWC → CHW for PyTorch
    model_input = np.transpose(normalized, (2, 0, 1))

    return model_input, card_image, info
