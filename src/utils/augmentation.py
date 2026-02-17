"""
PokéGrader — Data Augmentation

Simulates real-world phone camera conditions to expand training data.
Every augmentation is something that actually happens when someone
photographs a card with their phone:

    - Slight blur (hand shake, autofocus miss)
    - Lighting variation (warm lamp, cool daylight, dim room)
    - Color temperature shifts (phone white balance differences)
    - JPEG compression artifacts (phone camera compression)
    - Minor rotation (card not perfectly aligned in frame)
    - Brightness/contrast shifts (auto-exposure differences)
    - Slight perspective warp (phone not perfectly parallel to card)
    - Sensor noise (low light noise from phone cameras)
    - Shadow/vignette (uneven lighting on the card)

Each augmentation has realistic parameter ranges based on what phone
cameras actually produce. No extreme transforms that would never
happen in real usage.

Usage (standalone):
    from augmentation import Augmentor

    aug = Augmentor(seed=42)
    augmented = aug.augment(image)                    # Random augmentation
    augmented = aug.augment(image, intensity="heavy")  # More aggressive
    batch = aug.augment_batch(image, n=5)             # Multiple variants

Usage (with pipeline dataset):
    from augmentation import augment_dataset

    augment_dataset(
        source_dir="data/datasets/v001",
        output_dir="data/augmented/v001",
        multiplier=5,    # 5 augmented copies per original
    )

Usage (as PyTorch transform for training-time augmentation):
    from augmentation import PhoneCameraTransform

    transform = torchvision.transforms.Compose([
        PhoneCameraTransform(intensity="medium"),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225]),
    ])
"""

import cv2
import json
import numpy as np
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Augmentation parameters — calibrated to real phone camera behavior
# ---------------------------------------------------------------------------

@dataclass
class AugmentParams:
    """Parameter ranges for augmentations. All ranges are (min, max)."""

    # Gaussian blur (simulates slight hand shake / focus miss)
    blur_probability: float = 0.3
    blur_kernel_range: tuple[int, int] = (3, 7)

    # Brightness shift (auto-exposure variation)
    brightness_probability: float = 0.5
    brightness_range: tuple[float, float] = (-30, 30)

    # Contrast adjustment
    contrast_probability: float = 0.4
    contrast_range: tuple[float, float] = (0.8, 1.2)

    # Color temperature shift (white balance differences)
    color_temp_probability: float = 0.4
    color_temp_range: tuple[float, float] = (-15, 15)

    # Hue/saturation shift
    hue_probability: float = 0.3
    hue_range: tuple[int, int] = (-8, 8)
    saturation_range: tuple[float, float] = (0.85, 1.15)

    # JPEG compression artifacts
    jpeg_probability: float = 0.4
    jpeg_quality_range: tuple[int, int] = (60, 95)

    # Rotation (card not perfectly aligned)
    rotation_probability: float = 0.4
    rotation_range: tuple[float, float] = (-3.0, 3.0)  # degrees

    # Perspective warp (phone not parallel to card)
    perspective_probability: float = 0.3
    perspective_strength: float = 0.02

    # Gaussian noise (sensor noise in low light)
    noise_probability: float = 0.3
    noise_std_range: tuple[float, float] = (3, 12)

    # Vignette (uneven lighting / lens effect)
    vignette_probability: float = 0.2
    vignette_strength_range: tuple[float, float] = (0.1, 0.4)

    # Shadow gradient (light coming from one side)
    shadow_probability: float = 0.25
    shadow_strength_range: tuple[float, float] = (0.05, 0.20)


# Preset intensity levels
INTENSITY_PRESETS = {
    "light": AugmentParams(
        blur_probability=0.2,
        blur_kernel_range=(3, 5),
        brightness_probability=0.3,
        brightness_range=(-15, 15),
        contrast_probability=0.2,
        contrast_range=(0.9, 1.1),
        color_temp_probability=0.2,
        color_temp_range=(-8, 8),
        hue_probability=0.15,
        hue_range=(-5, 5),
        saturation_range=(0.9, 1.1),
        jpeg_probability=0.2,
        jpeg_quality_range=(75, 95),
        rotation_probability=0.2,
        rotation_range=(-1.5, 1.5),
        perspective_probability=0.15,
        perspective_strength=0.01,
        noise_probability=0.15,
        noise_std_range=(2, 6),
        vignette_probability=0.1,
        vignette_strength_range=(0.05, 0.2),
        shadow_probability=0.1,
        shadow_strength_range=(0.03, 0.10),
    ),
    "medium": AugmentParams(),  # defaults
    "heavy": AugmentParams(
        blur_probability=0.5,
        blur_kernel_range=(3, 9),
        brightness_probability=0.6,
        brightness_range=(-50, 50),
        contrast_probability=0.5,
        contrast_range=(0.7, 1.3),
        color_temp_probability=0.5,
        color_temp_range=(-25, 25),
        hue_probability=0.4,
        hue_range=(-12, 12),
        saturation_range=(0.75, 1.25),
        jpeg_probability=0.5,
        jpeg_quality_range=(40, 90),
        rotation_probability=0.5,
        rotation_range=(-5.0, 5.0),
        perspective_probability=0.4,
        perspective_strength=0.03,
        noise_probability=0.4,
        noise_std_range=(5, 20),
        vignette_probability=0.3,
        vignette_strength_range=(0.15, 0.5),
        shadow_probability=0.3,
        shadow_strength_range=(0.08, 0.25),
    ),
}


# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------

class Augmentor:
    """
    Applies realistic phone-camera augmentations to card images.

    Each augmentation simulates something that actually happens
    when photographing cards with a phone camera.
    """

    def __init__(
        self,
        params: Optional[AugmentParams] = None,
        intensity: str = "medium",
        seed: Optional[int] = None,
    ):
        """
        Args:
            params:    Custom augmentation parameters. Overrides intensity.
            intensity: Preset level: "light", "medium", or "heavy".
            seed:      Random seed for reproducibility.
        """
        if params is not None:
            self.params = params
        else:
            self.params = INTENSITY_PRESETS.get(intensity, INTENSITY_PRESETS["medium"])

        self.rng = np.random.RandomState(seed)

    def augment(
        self,
        image: np.ndarray,
        intensity: Optional[str] = None,
    ) -> np.ndarray:
        """
        Apply a random combination of augmentations to an image.

        Args:
            image:     BGR image (numpy array, uint8).
            intensity: Override the preset for this call only.

        Returns:
            Augmented image (same shape, uint8).
        """
        params = self.params
        if intensity is not None:
            params = INTENSITY_PRESETS.get(intensity, self.params)

        img = image.copy()

        # Apply each augmentation with its probability
        if self.rng.random() < params.brightness_probability:
            img = self._brightness(img, params)

        if self.rng.random() < params.contrast_probability:
            img = self._contrast(img, params)

        if self.rng.random() < params.color_temp_probability:
            img = self._color_temperature(img, params)

        if self.rng.random() < params.hue_probability:
            img = self._hue_saturation(img, params)

        if self.rng.random() < params.shadow_probability:
            img = self._shadow_gradient(img, params)

        if self.rng.random() < params.vignette_probability:
            img = self._vignette(img, params)

        if self.rng.random() < params.blur_probability:
            img = self._blur(img, params)

        if self.rng.random() < params.noise_probability:
            img = self._gaussian_noise(img, params)

        if self.rng.random() < params.rotation_probability:
            img = self._rotation(img, params)

        if self.rng.random() < params.perspective_probability:
            img = self._perspective_warp(img, params)

        # JPEG compression last (it's a lossy encoding step)
        if self.rng.random() < params.jpeg_probability:
            img = self._jpeg_compression(img, params)

        return img

    def augment_batch(
        self,
        image: np.ndarray,
        n: int = 5,
        intensity: Optional[str] = None,
    ) -> list[np.ndarray]:
        """Generate n augmented variants of a single image."""
        return [self.augment(image, intensity=intensity) for _ in range(n)]

    # ------------------------------------------------------------------
    # Individual augmentations
    # ------------------------------------------------------------------

    def _blur(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Gaussian blur — simulates hand shake or slight focus miss."""
        ksize = self.rng.choice(range(p.blur_kernel_range[0], p.blur_kernel_range[1] + 1, 2))
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def _brightness(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Brightness shift — auto-exposure variation between phones."""
        delta = self.rng.uniform(*p.brightness_range)
        return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    def _contrast(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Contrast adjustment — different camera processing."""
        factor = self.rng.uniform(*p.contrast_range)
        mean = img.mean()
        adjusted = (img.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _color_temperature(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Color temperature shift — warm (incandescent) vs cool (daylight)."""
        shift = self.rng.uniform(*p.color_temp_range)
        result = img.astype(np.float32)
        # Warm = boost red/yellow, Cool = boost blue
        result[:, :, 2] = np.clip(result[:, :, 2] + shift, 0, 255)       # Red
        result[:, :, 0] = np.clip(result[:, :, 0] - shift * 0.5, 0, 255) # Blue
        return result.astype(np.uint8)

    def _hue_saturation(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Hue/saturation shift — phone camera color processing differences."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.rng.uniform(*p.hue_range)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.rng.uniform(*p.saturation_range), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _jpeg_compression(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """JPEG artifacts — every phone compresses photos."""
        quality = self.rng.randint(*p.jpeg_quality_range)
        _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _rotation(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Slight rotation — card not perfectly aligned when photographed."""
        angle = self.rng.uniform(*p.rotation_range)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h),
                              borderMode=cv2.BORDER_REFLECT_101)

    def _perspective_warp(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Subtle perspective warp — phone not perfectly parallel to card."""
        h, w = img.shape[:2]
        strength = p.perspective_strength

        # Random offsets for each corner
        offsets = self.rng.uniform(-strength, strength, (4, 2))
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = src + (offsets * np.array([w, h])).astype(np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, matrix, (w, h),
                                   borderMode=cv2.BORDER_REFLECT_101)

    def _gaussian_noise(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Sensor noise — especially in low light phone photography."""
        std = self.rng.uniform(*p.noise_std_range)
        noise = self.rng.randn(*img.shape) * std
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _vignette(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Vignette — lens darkening at edges, common on phone cameras."""
        h, w = img.shape[:2]
        strength = self.rng.uniform(*p.vignette_strength_range)

        # Create radial gradient from center
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = np.sqrt(cx ** 2 + cy ** 2)
        mask = 1.0 - strength * (radius / max_radius) ** 2

        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= mask
        return np.clip(result, 0, 255).astype(np.uint8)

    def _shadow_gradient(self, img: np.ndarray, p: AugmentParams) -> np.ndarray:
        """Directional shadow — light coming from one side."""
        h, w = img.shape[:2]
        strength = self.rng.uniform(*p.shadow_strength_range)

        # Pick a random direction
        direction = self.rng.choice(["left", "right", "top", "bottom"])

        if direction == "left":
            gradient = np.linspace(1.0 - strength, 1.0, w)
            mask = np.tile(gradient, (h, 1))
        elif direction == "right":
            gradient = np.linspace(1.0, 1.0 - strength, w)
            mask = np.tile(gradient, (h, 1))
        elif direction == "top":
            gradient = np.linspace(1.0 - strength, 1.0, h)
            mask = np.tile(gradient.reshape(-1, 1), (1, w))
        else:
            gradient = np.linspace(1.0, 1.0 - strength, h)
            mask = np.tile(gradient.reshape(-1, 1), (1, w))

        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= mask
        return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# PyTorch-compatible transform
# ---------------------------------------------------------------------------

class PhoneCameraTransform:
    """
    Use as a torchvision transform for training-time augmentation.

    Example:
        transform = transforms.Compose([
            PhoneCameraTransform(intensity="medium"),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    """

    def __init__(self, intensity: str = "medium", seed: Optional[int] = None):
        self.augmentor = Augmentor(intensity=intensity, seed=seed)

    def __call__(self, image):
        """
        Args:
            image: PIL Image or numpy array.

        Returns:
            Augmented PIL Image (or numpy array if input was numpy).
        """
        # Handle PIL Image input
        is_pil = False
        try:
            from PIL import Image
            if isinstance(image, Image.Image):
                is_pil = True
                img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        if not is_pil:
            img_array = image

        augmented = self.augmentor.augment(img_array)

        if is_pil:
            from PIL import Image
            return Image.fromarray(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))

        return augmented


# ---------------------------------------------------------------------------
# Dataset augmentation (offline, expand dataset on disk)
# ---------------------------------------------------------------------------

def augment_dataset(
    source_dir: str,
    output_dir: Optional[str] = None,
    multiplier: int = 5,
    intensity: str = "medium",
    include_original: bool = True,
    seed: int = 42,
):
    """
    Augment an entire training dataset on disk.

    Takes a dataset directory (from pipeline.build_dataset()) and creates
    augmented copies of every image. Preserves the directory structure
    (train/val/test splits and grade folders).

    Args:
        source_dir:       Path to dataset (e.g. "data/datasets/v001").
        output_dir:       Output path. Defaults to data/augmented/{version}.
        multiplier:       Number of augmented copies per original image.
        intensity:        Augmentation intensity ("light", "medium", "heavy").
        include_original: Also copy the unaugmented original.
        seed:             Random seed for reproducibility.

    Note:
        Only augments the training split. Validation and test sets are
        copied as-is to keep evaluation clean.
    """
    src = Path(source_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if output_dir is None:
        version = src.name
        output_dir = str(src.parent.parent / "augmented" / version)

    dst = Path(output_dir)
    aug = Augmentor(intensity=intensity, seed=seed)

    stats = {"original": 0, "augmented": 0, "val_test_copied": 0}

    for split in ["train", "val", "test"]:
        split_src = src / split
        if not split_src.exists():
            continue

        for grade_dir in sorted(split_src.iterdir()):
            if not grade_dir.is_dir():
                continue

            dst_grade = dst / split / grade_dir.name
            dst_grade.mkdir(parents=True, exist_ok=True)

            images = list(grade_dir.glob("*.jpg")) + list(grade_dir.glob("*.png"))

            for img_path in images:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                stem = img_path.stem

                # Only augment training data — val/test stay clean
                if split == "train":
                    if include_original:
                        shutil.copy2(img_path, dst_grade / img_path.name)
                        stats["original"] += 1

                    for i in range(multiplier):
                        augmented = aug.augment(image)
                        out_name = f"{stem}_aug{i:02d}.jpg"
                        cv2.imwrite(
                            str(dst_grade / out_name),
                            augmented,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        stats["augmented"] += 1
                else:
                    # Copy val/test as-is
                    shutil.copy2(img_path, dst_grade / img_path.name)
                    stats["val_test_copied"] += 1

    # Write augmentation manifest
    manifest = {
        "source": str(src),
        "intensity": intensity,
        "multiplier": multiplier,
        "include_original": include_original,
        "seed": seed,
        "stats": stats,
    }
    with open(dst / "augmentation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = stats["original"] + stats["augmented"]
    print(f"  ✓ Augmented dataset → {dst}")
    print(f"    Train: {stats['original']} originals + {stats['augmented']} augmented = {total}")
    print(f"    Val/Test: {stats['val_test_copied']} copied as-is")

    return dst


# ---------------------------------------------------------------------------
# Preview utility — generates a visual comparison grid
# ---------------------------------------------------------------------------

def preview_augmentations(
    image_path: str,
    output_path: Optional[str] = None,
    n: int = 8,
    intensity: str = "medium",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a visual grid showing original + augmented variants.
    Useful for tuning parameters and verifying augmentations look realistic.

    Args:
        image_path:  Path to a card image.
        output_path: Save grid here (optional).
        n:           Number of augmented variants to show.
        intensity:   Augmentation intensity level.
        seed:        Random seed.

    Returns:
        Grid image as numpy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    aug = Augmentor(intensity=intensity, seed=seed)

    # Resize all to consistent size for grid
    thumb_h, thumb_w = 350, 250
    images = [cv2.resize(image, (thumb_w, thumb_h))]

    for i in range(n):
        variant = aug.augment(image)
        images.append(cv2.resize(variant, (thumb_w, thumb_h)))

    # Add labels
    labeled = []
    labels = ["Original"] + [f"Aug #{i+1}" for i in range(n)]
    for img, label in zip(images, labels):
        canvas = img.copy()
        # Dark overlay for text
        cv2.rectangle(canvas, (0, 0), (thumb_w, 28), (0, 0, 0), -1)
        cv2.putText(canvas, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        labeled.append(canvas)

    # Arrange in grid
    cols = min(4, len(labeled))
    rows = (len(labeled) + cols - 1) // cols

    # Pad to fill grid
    while len(labeled) < rows * cols:
        labeled.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_imgs = labeled[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))

    grid = np.vstack(grid_rows)

    if output_path:
        cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  ✓ Preview grid saved → {output_path}")

    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="PokéGrader Data Augmentation")
    sub = parser.add_subparsers(dest="command")

    # augment-dataset
    ad = sub.add_parser("dataset", help="Augment an entire dataset")
    ad.add_argument("source", help="Source dataset directory")
    ad.add_argument("--output", help="Output directory")
    ad.add_argument("--multiplier", type=int, default=5)
    ad.add_argument("--intensity", choices=["light", "medium", "heavy"], default="medium")
    ad.add_argument("--seed", type=int, default=42)

    # preview
    pv = sub.add_parser("preview", help="Generate augmentation preview grid")
    pv.add_argument("image", help="Path to a card image")
    pv.add_argument("--output", default="augmentation_preview.jpg")
    pv.add_argument("--n", type=int, default=8)
    pv.add_argument("--intensity", choices=["light", "medium", "heavy"], default="medium")

    # single
    sg = sub.add_parser("single", help="Augment a single image")
    sg.add_argument("image", help="Input image path")
    sg.add_argument("output", help="Output image path")
    sg.add_argument("--intensity", choices=["light", "medium", "heavy"], default="medium")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "dataset":
        augment_dataset(
            args.source,
            output_dir=args.output,
            multiplier=args.multiplier,
            intensity=args.intensity,
            seed=args.seed,
        )

    elif args.command == "preview":
        preview_augmentations(
            args.image,
            output_path=args.output,
            n=args.n,
            intensity=args.intensity,
        )

    elif args.command == "single":
        image = cv2.imread(args.image)
        if image is None:
            print(f"Could not read: {args.image}")
            return
        aug = Augmentor(intensity=args.intensity)
        result = aug.augment(image)
        cv2.imwrite(args.output, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  ✓ Augmented → {args.output}")


if __name__ == "__main__":
    _cli()
