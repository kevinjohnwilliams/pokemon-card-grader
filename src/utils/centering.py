"""
PokéGrader — Centering Analysis

Measures the centering of a Pokémon card by analyzing border symmetry.
This is purely algorithmic — no ML model needed.

PSA Centering Standards (approximate):
    10 (Gem Mint):   60/40 or better on front, 75/25 or better on back
    9  (Mint):       65/35 or better on front, 90/10 or better on back
    8  (NM-MT):      70/30 or better on front, 90/10 or better on back
    7  (Near Mint):   75/25 or better on front, 90/10 or better on back

We analyze the front only (user would need to flip for back).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CenteringResult:
    """Result of centering analysis."""
    score: float  # 0.0 to 1.0 (1.0 = perfect centering)
    grade_estimate: int  # PSA grade estimate for centering only
    left_right_ratio: str  # e.g., "52/48"
    top_bottom_ratio: str  # e.g., "55/45"
    left_border: float  # in pixels
    right_border: float
    top_border: float
    bottom_border: float
    horizontal_offset_pct: float  # How far off center horizontally (0 = perfect)
    vertical_offset_pct: float  # How far off center vertically (0 = perfect)
    details: str  # Human-readable description


def analyze_centering(card_image: np.ndarray) -> CenteringResult:
    """
    Analyze the centering of a cropped, perspective-corrected card image.

    Measures the border width on all four sides by detecting where the
    card artwork/frame begins. Works by finding the inner border of
    the card (the colored border around the artwork).

    Args:
        card_image: Cropped card image (BGR, portrait orientation)

    Returns:
        CenteringResult with scores and measurements
    """
    h, w = card_image.shape[:2]

    # Convert to multiple color spaces for robust edge detection
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

    # Step 1: Find the card border edges
    # The yellow/colored border of a Pokémon card has distinct color
    # We look for the transition from border to inner frame

    borders = _measure_borders(gray, h, w)
    left_b, right_b, top_b, bottom_b = borders

    # Step 2: Calculate ratios
    h_total = left_b + right_b
    v_total = top_b + bottom_b

    if h_total < 1:
        h_total = 1
    if v_total < 1:
        v_total = 1

    left_pct = (left_b / h_total) * 100
    right_pct = (right_b / h_total) * 100
    top_pct = (top_b / v_total) * 100
    bottom_pct = (bottom_b / v_total) * 100

    # Normalize to smaller/larger format (e.g., 55/45)
    lr_small = min(left_pct, right_pct)
    lr_large = max(left_pct, right_pct)
    tb_small = min(top_pct, bottom_pct)
    tb_large = max(top_pct, bottom_pct)

    lr_ratio = f"{lr_small:.0f}/{lr_large:.0f}"
    tb_ratio = f"{tb_small:.0f}/{tb_large:.0f}"

    # Step 3: Calculate centering score
    # Perfect centering = 50/50 on both axes
    # We measure deviation from perfect
    h_offset = abs(left_pct - 50)  # 0 = perfect, 50 = worst
    v_offset = abs(top_pct - 50)

    # Combined score (0-1, higher is better)
    # Max offset is 50 (all border on one side)
    h_score = 1.0 - (h_offset / 50)
    v_score = 1.0 - (v_offset / 50)

    # Weight horizontal and vertical equally
    score = (h_score + v_score) / 2

    # Step 4: Estimate PSA centering grade
    grade = _score_to_grade(lr_small, tb_small)

    # Step 5: Human-readable description
    details = _describe_centering(lr_small, lr_large, tb_small, tb_large, grade)

    return CenteringResult(
        score=round(score, 4),
        grade_estimate=grade,
        left_right_ratio=lr_ratio,
        top_bottom_ratio=tb_ratio,
        left_border=round(left_b, 1),
        right_border=round(right_b, 1),
        top_border=round(top_b, 1),
        bottom_border=round(bottom_b, 1),
        horizontal_offset_pct=round(h_offset, 2),
        vertical_offset_pct=round(v_offset, 2),
        details=details,
    )


def _measure_borders(gray: np.ndarray, h: int, w: int) -> tuple[float, float, float, float]:
    """
    Measure border widths by finding where the card content begins
    on each side using gradient analysis.
    """
    # Use Sobel gradients to find strong edges (border → content transitions)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)

    # Measure left border: scan from left, find first strong vertical edge
    # Sample the middle 60% of the card height to avoid corners
    y_start = int(h * 0.2)
    y_end = int(h * 0.8)

    left_border = _find_edge_from_side(abs_sobel_x[y_start:y_end, :], axis=1, from_start=True)
    right_border = _find_edge_from_side(abs_sobel_x[y_start:y_end, :], axis=1, from_start=False)

    # Measure top/bottom: sample middle 60% of width
    x_start = int(w * 0.2)
    x_end = int(w * 0.8)

    top_border = _find_edge_from_side(abs_sobel_y[:, x_start:x_end], axis=0, from_start=True)
    bottom_border = _find_edge_from_side(abs_sobel_y[:, x_start:x_end], axis=0, from_start=False)

    # Sanity check — borders shouldn't be more than 15% of the card dimension
    max_h_border = w * 0.15
    max_v_border = h * 0.15

    left_border = min(left_border, max_h_border)
    right_border = min(right_border, max_h_border)
    top_border = min(top_border, max_v_border)
    bottom_border = min(bottom_border, max_v_border)

    return left_border, right_border, top_border, bottom_border


def _find_edge_from_side(
    gradient: np.ndarray,
    axis: int,
    from_start: bool,
    threshold_ratio: float = 0.3,
) -> float:
    """
    Find the first significant edge by scanning from one side.

    Args:
        gradient: Absolute gradient values (2D array)
        axis: 0 for vertical scan (top/bottom), 1 for horizontal scan (left/right)
        from_start: True to scan from start, False to scan from end
        threshold_ratio: Edge detection sensitivity

    Returns:
        Border width in pixels
    """
    # Sum gradient along the perpendicular axis to get a 1D profile
    profile = np.mean(gradient, axis=(1 - axis))

    if not from_start:
        profile = profile[::-1]

    # Smooth the profile
    kernel_size = max(3, len(profile) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(profile.reshape(-1, 1), (1, kernel_size), 0).flatten()

    # Find threshold
    max_val = np.max(smoothed)
    if max_val < 1:
        return 0.0

    threshold = max_val * threshold_ratio

    # Find first crossing above threshold (skip the very edge pixels)
    skip = max(2, len(smoothed) // 40)
    for i in range(skip, len(smoothed)):
        if smoothed[i] > threshold:
            return float(i)

    return 0.0


def _score_to_grade(lr_small: float, tb_small: float) -> int:
    """
    Convert centering ratios to a PSA-style grade (centering component only).

    Based on PSA centering standards:
        10: 60/40 or better (smallest side >= 40%)
        9:  65/35 or better (smallest side >= 35%)
        8:  70/30 or better (smallest side >= 30%)
        7:  75/25 or better (smallest side >= 25%)
        6:  80/20 or better (smallest side >= 20%)
        5:  85/15 or better
        4:  88/12 or better
        3:  90/10 or better
    """
    # Use the worse of the two axes
    worst = min(lr_small, tb_small)

    if worst >= 40:
        return 10
    elif worst >= 35:
        return 9
    elif worst >= 30:
        return 8
    elif worst >= 25:
        return 7
    elif worst >= 20:
        return 6
    elif worst >= 15:
        return 5
    elif worst >= 12:
        return 4
    elif worst >= 10:
        return 3
    elif worst >= 5:
        return 2
    else:
        return 1


def _describe_centering(
    lr_small: float, lr_large: float,
    tb_small: float, tb_large: float,
    grade: int,
) -> str:
    """Generate a human-readable centering description."""
    descriptions = {
        10: "Exceptional centering — virtually perfect alignment.",
        9: "Excellent centering with only slight deviation.",
        8: "Very good centering — minor shift visible on close inspection.",
        7: "Noticeable off-center but within acceptable range.",
        6: "Moderately off-center — visible without close inspection.",
        5: "Significantly off-center on at least one axis.",
        4: "Substantially off-center.",
        3: "Severely off-center.",
        2: "Extremely off-center.",
        1: "Card is almost entirely shifted to one side.",
    }

    base = descriptions.get(grade, "")
    lr_note = f"L/R: {lr_small:.0f}/{lr_large:.0f}"
    tb_note = f"T/B: {tb_small:.0f}/{tb_large:.0f}"

    return f"{base} ({lr_note}, {tb_note})"
