"""
PokéGrader — Card Detection & Cropping

Uses OpenCV to detect a Pokémon card in a camera photo,
apply perspective correction, and crop it out cleanly.

Works with any background — finds the largest rectangular
contour that matches card proportions (2.5 x 3.5 inches).
"""

import cv2
import numpy as np
from typing import Optional


# Standard Pokémon card aspect ratio (2.5" x 3.5")
CARD_ASPECT_RATIO = 2.5 / 3.5  # ~0.714
ASPECT_TOLERANCE = 0.15  # Allow some deviation

# Output dimensions for cropped card (maintaining aspect ratio)
OUTPUT_WIDTH = 500
OUTPUT_HEIGHT = int(OUTPUT_WIDTH / CARD_ASPECT_RATIO)  # ~700


def detect_and_crop_card(
    image: np.ndarray,
    debug: bool = False,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Detect a card in the image, perspective-correct it, and crop.

    Args:
        image: BGR image from camera/upload (numpy array)
        debug: If True, return annotated debug image

    Returns:
        Tuple of:
            - cropped: Perspective-corrected card image (or None if not found)
            - debug_img: Annotated image showing detection (or None)
            - info: Dict with detection metadata
    """
    info = {"detected": False, "confidence": 0.0, "corners": None}
    debug_img = image.copy() if debug else None

    # Resize for faster processing (keep aspect ratio)
    h_orig, w_orig = image.shape[:2]
    scale = 800 / max(h_orig, w_orig)
    if scale < 1:
        working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        working = image.copy()
        scale = 1.0

    h_work, w_work = working.shape[:2]

    # Step 1: Preprocess — grayscale, blur, edge detection
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding handles varying lighting
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Step 2: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, debug_img, info

    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_corners = None

    # Step 3: Find the best quadrilateral that matches card proportions
    for contour in contours[:10]:  # Check top 10 largest contours
        area = cv2.contourArea(contour)

        # Skip very small contours (less than 5% of image area)
        if area < (h_work * w_work * 0.05):
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # We want exactly 4 corners (a quadrilateral)
        if len(approx) == 4:
            # Check if the aspect ratio roughly matches a card
            corners = approx.reshape(4, 2).astype(np.float32)
            if _is_card_shaped(corners):
                card_corners = corners
                info["confidence"] = min(area / (h_work * w_work), 1.0)
                break

    # Step 4: If no 4-corner contour found, try convex hull approach
    if card_corners is None:
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < (h_work * w_work * 0.05):
                continue

            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                if _is_card_shaped(corners):
                    card_corners = corners
                    info["confidence"] = min(area / (h_work * w_work), 1.0) * 0.8
                    break

    if card_corners is None:
        return None, debug_img, info

    # Step 5: Scale corners back to original image size
    card_corners_orig = card_corners / scale

    # Order corners: top-left, top-right, bottom-right, bottom-left
    ordered = _order_corners(card_corners_orig)
    info["corners"] = ordered.tolist()
    info["detected"] = True

    # Step 6: Perspective transform
    dst_pts = np.array([
        [0, 0],
        [OUTPUT_WIDTH - 1, 0],
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
        [0, OUTPUT_HEIGHT - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
    cropped = cv2.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # Draw debug overlay
    if debug and debug_img is not None:
        pts = ordered.astype(np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 255, 255), 3)
        for i, pt in enumerate(pts):
            cv2.circle(debug_img, tuple(pt), 8, (0, 200, 255), -1)
            cv2.putText(debug_img, str(i), tuple(pt + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return cropped, debug_img, info


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corners as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference: smallest = top-right, largest = bottom-left
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def _is_card_shaped(corners: np.ndarray) -> bool:
    """
    Check if a quadrilateral has roughly the proportions of a card.
    """
    ordered = _order_corners(corners)

    # Calculate width and height from corners
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])

    avg_w = (w_top + w_bot) / 2
    avg_h = (h_left + h_right) / 2

    if avg_h < 1:
        return False

    aspect = avg_w / avg_h

    # Check if aspect ratio is close to a card (portrait or landscape)
    portrait_match = abs(aspect - CARD_ASPECT_RATIO) < ASPECT_TOLERANCE
    landscape_match = abs(aspect - (1 / CARD_ASPECT_RATIO)) < ASPECT_TOLERANCE

    return portrait_match or landscape_match


def ensure_portrait(card_image: np.ndarray) -> np.ndarray:
    """
    Ensure the card image is in portrait orientation (taller than wide).
    Rotates 90° if needed.
    """
    h, w = card_image.shape[:2]
    if w > h:
        return cv2.rotate(card_image, cv2.ROTATE_90_CLOCKWISE)
    return card_image
