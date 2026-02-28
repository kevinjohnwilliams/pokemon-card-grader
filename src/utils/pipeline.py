"""
PokéGrader — Training Pipeline

End-to-end pipeline for building training data through real PSA submissions.
Photographs taken with the app (phone camera) are paired with PSA grades
when results return, creating labeled training data with zero domain gap.

Flow:
    capture → predict → submit → receive → dataset → train → deploy → repeat

Usage (Python):
    from pipeline import Pipeline

    p = Pipeline("./data")

    card_id = p.capture("photo.jpg")
    p.predict(card_id, grade=8, model_version="v0.1")
    p.submit("PSA-2025-001", [card_id], tier="bulk")

    # When PSA returns...
    p.receive("PSA-2025-001", {card_id: 7})

    # Build dataset for training
    p.build_dataset()
    p.status()

Usage (CLI):
    python pipeline.py capture photo.jpg
    python pipeline.py capture photo.jpg --id my-charizard
    python pipeline.py predict abc123 8 --model v0.1
    python pipeline.py submit PSA-2025-001 abc123 def456 --tier bulk
    python pipeline.py receive abc123 7
    python pipeline.py receive-batch PSA-2025-001 abc123:7 def456:9
    python pipeline.py dataset
    python pipeline.py status
    python pipeline.py list --filter pending
    python pipeline.py export-csv
"""

import json
import uuid
import shutil
import csv
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """A single card flowing through the pipeline."""

    card_id: str
    raw_path: str                            # Original phone photo
    crop_path: Optional[str] = None          # Detected & cropped card
    captured_at: str = ""

    # App prediction (logged before PSA submission)
    predicted_grade: Optional[int] = None
    centering_score: Optional[float] = None
    centering_lr: Optional[str] = None       # e.g. "55/45"
    centering_tb: Optional[str] = None
    model_version: Optional[str] = None
    confidence: Optional[float] = None

    # PSA submission tracking
    submission_id: Optional[str] = None
    submitted_at: Optional[str] = None
    service_tier: Optional[str] = None

    # Ground truth
    psa_grade: Optional[int] = None
    psa_subgrades: Optional[dict] = None     # If PSA provides sub-grades
    received_at: Optional[str] = None

    # Computed after receiving grade
    grade_delta: Optional[int] = None        # predicted - actual

    # Dataset assignment
    dataset_version: Optional[str] = None
    split: Optional[str] = None              # train / val / test

    # Optional metadata
    card_name: Optional[str] = None          # e.g. "Charizard VMAX"
    set_name: Optional[str] = None           # e.g. "Darkness Ablaze"
    notes: Optional[str] = None

    @property
    def is_labeled(self) -> bool:
        return self.psa_grade is not None

    @property
    def is_submitted(self) -> bool:
        return self.submission_id is not None

    @property
    def is_pending(self) -> bool:
        return self.is_submitted and not self.is_labeled

    @property
    def has_prediction(self) -> bool:
        return self.predicted_grade is not None

    @property
    def has_crop(self) -> bool:
        return self.crop_path is not None and Path(self.crop_path).exists()


@dataclass
class Submission:
    """A batch PSA submission."""

    submission_id: str
    card_ids: list[str]
    tier: str
    submitted_at: str
    expected_return: Optional[str] = None
    cost: Optional[float] = None
    received: bool = False
    received_at: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class DatasetManifest:
    """Metadata for a versioned training dataset."""

    version: str
    created_at: str
    source_cards: int
    splits: dict[str, int]
    grade_distribution: dict[str, int]
    model_version_at_build: Optional[str] = None


# ---------------------------------------------------------------------------
# JSON database
# ---------------------------------------------------------------------------

class _Database:
    """Simple JSON-file persistence for cards and submissions."""

    def __init__(self, data_dir: Path):
        self._cards_path = data_dir / "cards.json"
        self._subs_path = data_dir / "submissions.json"
        self.cards: dict[str, Card] = {}
        self.submissions: dict[str, Submission] = {}
        self._load()

    def _load(self):
        if self._cards_path.exists():
            with open(self._cards_path) as f:
                raw = json.load(f)
            self.cards = {k: Card(**v) for k, v in raw.items()}

        if self._subs_path.exists():
            with open(self._subs_path) as f:
                raw = json.load(f)
            self.submissions = {k: Submission(**v) for k, v in raw.items()}

    def save(self):
        with open(self._cards_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.cards.items()}, f, indent=2)
        with open(self._subs_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.submissions.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Manages the full capture → PSA → train feedback loop.

    Directory layout:
        data/
        ├── cards/
        │   ├── raw/              # Original phone photos
        │   └── cropped/          # Detected & cropped cards
        ├── datasets/
        │   └── v001/
        │       ├── train/
        │       │   ├── grade_01/ ... grade_10/
        │       ├── val/
        │       └── test/
        ├── models/               # Saved checkpoints
        ├── cards.json            # Card records
        └── submissions.json      # PSA submission tracking
    """

    def __init__(self, data_dir: str = "./data"):
        self.root = Path(data_dir)
        self._ensure_dirs()
        self.db = _Database(self.root)

    def _ensure_dirs(self):
        for sub in ["cards/raw", "cards/cropped", "datasets", "models"]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: CAPTURE
    # ------------------------------------------------------------------

    def capture(
        self,
        image_path: str,
        card_id: Optional[str] = None,
        card_name: Optional[str] = None,
        set_name: Optional[str] = None,
        auto_crop: bool = True,
        run_centering: bool = True,
    ) -> str:
        """
        Register a card photo in the pipeline.

        Copies the photo into managed storage, optionally runs card
        detection to crop it, and optionally runs centering analysis.

        Args:
            image_path:    Path to the phone photo.
            card_id:       Custom ID. Auto-generated if omitted.
            card_name:     e.g. "Charizard VMAX"
            set_name:      e.g. "Darkness Ablaze"
            auto_crop:     Run card detection and save cropped image.
            run_centering: Run centering analysis on the crop.

        Returns:
            card_id
        """
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        card_id = card_id or uuid.uuid4().hex[:8]
        if card_id in self.db.cards:
            raise ValueError(f"Card ID '{card_id}' already exists")

        ext = src.suffix.lower() or ".jpg"
        raw_dest = self.root / "cards" / "raw" / f"{card_id}{ext}"
        shutil.copy2(src, raw_dest)

        card = Card(
            card_id=card_id,
            raw_path=str(raw_dest),
            captured_at=datetime.now().isoformat(),
            card_name=card_name,
            set_name=set_name,
        )

        # Auto-crop
        if auto_crop:
            crop = self._detect_and_crop(raw_dest, card_id)
            if crop:
                card.crop_path = str(crop)

                # Centering analysis on the crop
                if run_centering and crop:
                    self._run_centering(card, crop)

        self.db.cards[card_id] = card
        self.db.save()

        _log(f"Captured {card_id}" + (f" ({card_name})" if card_name else ""))
        return card_id

    def capture_batch(self, image_paths: list[str], **kwargs) -> list[str]:
        """Capture multiple cards at once. Returns list of card_ids."""
        ids = []
        for path in image_paths:
            try:
                cid = self.capture(path, **kwargs)
                ids.append(cid)
            except Exception as e:
                _log(f"Failed to capture {path}: {e}", error=True)
        return ids

    def _detect_and_crop(self, image_path: Path, card_id: str) -> Optional[Path]:
        """Run card detection, save crop. Returns crop path or None."""
        try:
            from card_detector import detect_and_crop_card, ensure_portrait
        except ImportError:
            try:
                from src.utils.card_detector import detect_and_crop_card, ensure_portrait
            except ImportError:
                _log("card_detector not importable — skipping auto-crop", error=True)
                return None

        image = cv2.imread(str(image_path))
        if image is None:
            _log(f"Could not read image: {image_path}", error=True)
            return None

        cropped, _, info = detect_and_crop_card(image)
        if cropped is None:
            _log(f"No card detected in {card_id}")
            return None

        cropped = ensure_portrait(cropped)
        dest = self.root / "cards" / "cropped" / f"{card_id}.jpg"
        cv2.imwrite(str(dest), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return dest

    def _run_centering(self, card: Card, crop_path: Path):
        """Run centering analysis and store results on the card record."""
        try:
            from centering import analyze_centering
        except ImportError:
            try:
                from src.utils.centering import analyze_centering
            except ImportError:
                return

        image = cv2.imread(str(crop_path))
        if image is None:
            return

        result = analyze_centering(image)
        card.centering_score = result.score
        card.centering_lr = result.left_right_ratio
        card.centering_tb = result.top_bottom_ratio

    # ------------------------------------------------------------------
    # Stage 2: PREDICT
    # ------------------------------------------------------------------

    def predict(
        self,
        card_id: str,
        grade: int,
        model_version: str = "v0.0-manual",
        confidence: Optional[float] = None,
    ):
        """
        Log a grade prediction BEFORE submitting to PSA.

        This is the key measurement — we compare this prediction
        against the real PSA grade to track model accuracy over time.
        """
        card = self._get_card(card_id)

        if not 1 <= grade <= 10:
            raise ValueError("Grade must be 1-10")

        card.predicted_grade = grade
        card.model_version = model_version
        card.confidence = confidence
        self.db.save()

        _log(f"Predicted {card_id} → PSA {grade} (model: {model_version})")

    # ------------------------------------------------------------------
    # Stage 3: SUBMIT
    # ------------------------------------------------------------------

    def submit(
        self,
        submission_id: str,
        card_ids: list[str],
        tier: str = "bulk",
        cost: Optional[float] = None,
        notes: Optional[str] = None,
    ):
        """
        Record a PSA submission batch.

        Args:
            submission_id:  Your PSA order number.
            card_ids:       Cards being submitted.
            tier:           Service level (bulk, regular, express, super_express, walk_through).
            cost:           Total submission cost.
        """
        if submission_id in self.db.submissions:
            raise ValueError(f"Submission '{submission_id}' already exists")

        # Validate all card IDs exist
        for cid in card_ids:
            self._get_card(cid)  # Raises if not found

        now = datetime.now().isoformat()

        sub = Submission(
            submission_id=submission_id,
            card_ids=list(card_ids),
            tier=tier,
            submitted_at=now,
            cost=cost,
            notes=notes,
        )

        for cid in card_ids:
            card = self.db.cards[cid]
            card.submission_id = submission_id
            card.submitted_at = now
            card.service_tier = tier

        self.db.submissions[submission_id] = sub
        self.db.save()

        _log(f"Submission {submission_id}: {len(card_ids)} cards ({tier})")

    # ------------------------------------------------------------------
    # Stage 4: RECEIVE GRADES
    # ------------------------------------------------------------------

    def receive(self, card_id: str, psa_grade: int, subgrades: Optional[dict] = None):
        """
        Record a single PSA grade result.

        Creates a labeled training pair: phone photo → real PSA grade.
        """
        card = self._get_card(card_id)

        if not 1 <= psa_grade <= 10:
            raise ValueError("Grade must be 1-10")

        card.psa_grade = psa_grade
        card.psa_subgrades = subgrades
        card.received_at = datetime.now().isoformat()

        if card.predicted_grade is not None:
            card.grade_delta = card.predicted_grade - psa_grade

        self.db.save()

        delta = f" (delta: {card.grade_delta:+d})" if card.grade_delta is not None else ""
        _log(f"Received {card_id} → PSA {psa_grade}{delta}")

    def receive_batch(self, submission_id: str, grades: dict[str, int]):
        """
        Record grades for an entire PSA submission.

        Args:
            submission_id:  The PSA submission ID.
            grades:         {card_id: psa_grade} mapping.
        """
        sub = self.db.submissions.get(submission_id)
        if not sub:
            raise ValueError(f"Submission '{submission_id}' not found")

        for card_id, grade in grades.items():
            self.receive(card_id, grade)

        sub.received = True
        sub.received_at = datetime.now().isoformat()
        self.db.save()

        _log(f"Submission {submission_id} complete: {len(grades)} grades")

    # ------------------------------------------------------------------
    # Stage 5: BUILD DATASET
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        version: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_cards: int = 30,
        seed: int = 42,
    ) -> Optional[Path]:
        """
        Build a versioned training dataset from labeled cards.

        Copies cropped card images into a directory structure:
            datasets/v001/train/grade_07/abc123.jpg

        Args:
            version:     Dataset version string. Auto-incremented if None.
            train_ratio: Fraction of data for training.
            val_ratio:   Fraction for validation.
            test_ratio:  Fraction for testing.
            min_cards:   Minimum labeled cards required to build.
            seed:        Random seed for reproducible splits.

        Returns:
            Path to dataset directory, or None if not enough data.
        """
        import random

        labeled = [c for c in self.db.cards.values() if c.is_labeled and c.has_crop]

        if len(labeled) < min_cards:
            _log(f"Only {len(labeled)} labeled cards — need {min_cards} minimum")
            return None

        # Auto-version
        if version is None:
            existing = sorted(self.root.joinpath("datasets").glob("v*"))
            n = len(existing) + 1
            version = f"v{n:03d}"

        ds_dir = self.root / "datasets" / version

        # Create grade directories for each split
        for split in ["train", "val", "test"]:
            for grade in range(1, 11):
                (ds_dir / split / f"grade_{grade:02d}").mkdir(parents=True, exist_ok=True)

        # Shuffle deterministically and split
        random.seed(seed)
        random.shuffle(labeled)

        n_test = max(1, int(len(labeled) * test_ratio))
        n_val = max(1, int(len(labeled) * val_ratio))

        splits = {
            "test": labeled[:n_test],
            "val": labeled[n_test : n_test + n_val],
            "train": labeled[n_test + n_val :],
        }

        # Copy images into dataset
        for split_name, cards in splits.items():
            for card in cards:
                card.split = split_name
                card.dataset_version = version
                dest = ds_dir / split_name / f"grade_{card.psa_grade:02d}" / f"{card.card_id}.jpg"
                shutil.copy2(card.crop_path, dest)

        self.db.save()

        # Write manifest
        manifest = DatasetManifest(
            version=version,
            created_at=datetime.now().isoformat(),
            source_cards=len(labeled),
            splits={k: len(v) for k, v in splits.items()},
            grade_distribution=_grade_dist(labeled),
        )
        with open(ds_dir / "manifest.json", "w") as f:
            json.dump(asdict(manifest), f, indent=2)

        _log(f"Dataset {version}: {len(labeled)} cards "
             f"(train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])})")
        self._print_grade_distribution(labeled)

        return ds_dir

    # ------------------------------------------------------------------
    # Status & queries
    # ------------------------------------------------------------------

    def status(self):
        """Print pipeline health dashboard."""
        cards = self.db.cards.values()
        total = len(self.db.cards)
        labeled = sum(1 for c in cards if c.is_labeled)
        pending = sum(1 for c in cards if c.is_pending)
        unsubmitted = sum(1 for c in cards if not c.is_submitted)
        cropped = sum(1 for c in cards if c.has_crop)

        print()
        print("=" * 52)
        print("  PokéGrader Pipeline")
        print("=" * 52)
        print(f"  📱 Total cards:              {total}")
        print(f"  ✂️  With crops:               {cropped}")
        print(f"  📋 Not submitted yet:        {unsubmitted}")
        print(f"  📦 Awaiting PSA results:     {pending}")
        print(f"  ✅ Labeled (training-ready):  {labeled}")
        print()

        # Grade distribution
        if labeled > 0:
            self._print_grade_distribution([c for c in cards if c.is_labeled])
            print()

        # Prediction accuracy
        self._print_accuracy()

        # Submissions
        active_subs = [s for s in self.db.submissions.values() if not s.received]
        if active_subs:
            print("  📦 Active Submissions:")
            for s in active_subs:
                print(f"     {s.submission_id}  |  {len(s.card_ids)} cards  |  {s.tier}  |  {s.submitted_at[:10]}")
            print()

        # Readiness
        if labeled >= 50:
            print("  🟢 Ready to train → pipeline.build_dataset()")
        elif labeled >= 25:
            print(f"  🟡 Getting close — {50 - labeled} more labeled cards to go")
        else:
            print(f"  🔴 Collecting data — {50 - labeled} more labeled cards needed")

        # Retrain check
        if self.should_retrain():
            print("  ⚡ New data available — consider retraining")

        print()

    def list_cards(self, filter_by: Optional[str] = None) -> list[Card]:
        """
        List cards with optional filter.

        Filters: all, pending, labeled, unsubmitted, no_crop
        """
        cards = list(self.db.cards.values())

        if filter_by == "pending":
            cards = [c for c in cards if c.is_pending]
        elif filter_by == "labeled":
            cards = [c for c in cards if c.is_labeled]
        elif filter_by == "unsubmitted":
            cards = [c for c in cards if not c.is_submitted]
        elif filter_by == "no_crop":
            cards = [c for c in cards if not c.has_crop]

        return cards

    def get_card(self, card_id: str) -> Card:
        """Get a card record by ID."""
        return self._get_card(card_id)

    def should_retrain(self, threshold: int = 25) -> bool:
        """Check if enough new labeled data exists since the last dataset build."""
        datasets = sorted(self.root.joinpath("datasets").glob("v*"))
        if not datasets:
            return len([c for c in self.db.cards.values() if c.is_labeled]) >= 50

        manifest_path = datasets[-1] / "manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        last_count = manifest.get("source_cards", 0)
        current_labeled = sum(1 for c in self.db.cards.values() if c.is_labeled)
        return (current_labeled - last_count) >= threshold

    def export_csv(self, output_path: Optional[str] = None) -> Path:
        """Export all card records to CSV for external analysis."""
        out = Path(output_path) if output_path else self.root / "cards_export.csv"

        fields = [
            "card_id", "card_name", "set_name", "captured_at",
            "centering_score", "centering_lr", "centering_tb",
            "predicted_grade", "model_version", "confidence",
            "submission_id", "service_tier", "submitted_at",
            "psa_grade", "received_at", "grade_delta",
            "dataset_version", "split",
        ]

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for card in self.db.cards.values():
                row = {k: getattr(card, k, None) for k in fields}
                writer.writerow(row)

        _log(f"Exported {len(self.db.cards)} cards → {out}")
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def recrop(self, card_id: str) -> bool:
        """Re-run card detection on an existing photo (e.g. after improving detector)."""
        card = self._get_card(card_id)
        raw = Path(card.raw_path)
        if not raw.exists():
            _log(f"Raw image missing for {card_id}", error=True)
            return False

        crop = self._detect_and_crop(raw, card_id)
        if crop:
            card.crop_path = str(crop)
            self._run_centering(card, crop)
            self.db.save()
            return True
        return False

    def delete_card(self, card_id: str, delete_files: bool = False):
        """Remove a card from the pipeline."""
        card = self._get_card(card_id)

        if delete_files:
            for path_str in [card.raw_path, card.crop_path]:
                if path_str:
                    p = Path(path_str)
                    if p.exists():
                        p.unlink()

        # Remove from any submission
        if card.submission_id and card.submission_id in self.db.submissions:
            sub = self.db.submissions[card.submission_id]
            sub.card_ids = [c for c in sub.card_ids if c != card_id]

        del self.db.cards[card_id]
        self.db.save()
        _log(f"Deleted {card_id}")

    def _get_card(self, card_id: str) -> Card:
        card = self.db.cards.get(card_id)
        if not card:
            raise ValueError(f"Card '{card_id}' not found")
        return card

    def _print_accuracy(self):
        paired = [c for c in self.db.cards.values()
                  if c.predicted_grade is not None and c.psa_grade is not None]
        if not paired:
            return

        exact = sum(1 for c in paired if c.grade_delta == 0)
        within_1 = sum(1 for c in paired if abs(c.grade_delta) <= 1)
        avg_delta = sum(c.grade_delta for c in paired) / len(paired)

        print(f"  🎯 Prediction Accuracy ({len(paired)} graded):")
        print(f"     Exact:    {exact}/{len(paired)} ({100 * exact / len(paired):.0f}%)")
        print(f"     Within ±1: {within_1}/{len(paired)} ({100 * within_1 / len(paired):.0f}%)")
        print(f"     Avg bias:  {avg_delta:+.2f}")
        print()

    def _print_grade_distribution(self, cards: list[Card]):
        dist = _grade_dist(cards)
        print("  📊 Grade Distribution:")
        for grade in range(1, 11):
            count = dist.get(str(grade), 0)
            if count > 0:
                bar = "█" * count + "░" * (max(dist.values()) - count)
                print(f"     PSA {grade:2d} │ {bar} {count}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grade_dist(cards: list[Card]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for c in cards:
        if c.psa_grade is not None:
            key = str(c.psa_grade)
            dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


def _log(msg: str, error: bool = False):
    prefix = "  ✗" if error else "  ✓"
    print(f"{prefix} {msg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="PokéGrader Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python pipeline.py capture photo.jpg
  python pipeline.py capture photo.jpg --id charizard-01 --name "Charizard VMAX"
  python pipeline.py predict abc123 8 --model v0.1
  python pipeline.py submit PSA-2025-001 abc123 def456 --tier bulk
  python pipeline.py receive abc123 7
  python pipeline.py receive-batch PSA-2025-001 abc123:7 def456:9
  python pipeline.py dataset
  python pipeline.py status
  python pipeline.py list --filter pending
  python pipeline.py export-csv
        """,
    )
    parser.add_argument("--data", default="./data", help="Data directory (default: ./data)")
    sub = parser.add_subparsers(dest="command")

    # capture
    cap = sub.add_parser("capture", help="Register a card photo")
    cap.add_argument("image", help="Path to phone photo")
    cap.add_argument("--id", help="Custom card ID")
    cap.add_argument("--name", help="Card name (e.g. 'Charizard VMAX')")
    cap.add_argument("--set", dest="set_name", help="Set name")
    cap.add_argument("--no-crop", action="store_true", help="Skip auto-crop")

    # predict
    pred = sub.add_parser("predict", help="Log a grade prediction")
    pred.add_argument("card_id")
    pred.add_argument("grade", type=int)
    pred.add_argument("--model", default="v0.0-manual")

    # submit
    sm = sub.add_parser("submit", help="Record a PSA submission")
    sm.add_argument("submission_id")
    sm.add_argument("card_ids", nargs="+")
    sm.add_argument("--tier", default="bulk")
    sm.add_argument("--cost", type=float)

    # receive (single)
    rv = sub.add_parser("receive", help="Record a PSA grade")
    rv.add_argument("card_id")
    rv.add_argument("grade", type=int)

    # receive-batch
    rb = sub.add_parser("receive-batch", help="Record grades for a submission (card_id:grade ...)")
    rb.add_argument("submission_id")
    rb.add_argument("grades", nargs="+", help="card_id:grade pairs")

    # dataset
    ds = sub.add_parser("dataset", help="Build training dataset")
    ds.add_argument("--min", type=int, default=30, help="Minimum cards required")

    # status
    sub.add_parser("status", help="Pipeline status dashboard")

    # list
    ls = sub.add_parser("list", help="List cards")
    ls.add_argument("--filter", choices=["all", "pending", "labeled", "unsubmitted", "no_crop"],
                     default="all")

    # export-csv
    ec = sub.add_parser("export-csv", help="Export cards to CSV")
    ec.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    p = Pipeline(args.data)

    if args.command == "capture":
        p.capture(args.image, card_id=args.id, card_name=args.name,
                  set_name=args.set_name, auto_crop=not args.no_crop)

    elif args.command == "predict":
        p.predict(args.card_id, args.grade, model_version=args.model)

    elif args.command == "submit":
        p.submit(args.submission_id, args.card_ids, tier=args.tier, cost=args.cost)

    elif args.command == "receive":
        p.receive(args.card_id, args.grade)

    elif args.command == "receive-batch":
        grades = {}
        for pair in args.grades:
            cid, g = pair.split(":")
            grades[cid] = int(g)
        p.receive_batch(args.submission_id, grades)

    elif args.command == "dataset":
        p.build_dataset(min_cards=args.min)

    elif args.command == "status":
        p.status()

    elif args.command == "list":
        cards = p.list_cards(filter_by=args.filter if args.filter != "all" else None)
        if not cards:
            print("  No cards found.")
            return
        # Table output
        print(f"\n  {'ID':<12} {'Name':<20} {'Predicted':<10} {'PSA':<6} {'Delta':<7} {'Status'}")
        print(f"  {'─' * 12} {'─' * 20} {'─' * 10} {'─' * 6} {'─' * 7} {'─' * 12}")
        for c in cards:
            name = (c.card_name or "—")[:20]
            pred = str(c.predicted_grade) if c.predicted_grade else "—"
            psa = str(c.psa_grade) if c.psa_grade else "—"
            delta = f"{c.grade_delta:+d}" if c.grade_delta is not None else "—"
            if c.is_labeled:
                status = "✅ labeled"
            elif c.is_pending:
                status = "📦 pending"
            elif c.is_submitted:
                status = "📤 submitted"
            else:
                status = "📱 captured"
            print(f"  {c.card_id:<12} {name:<20} {pred:<10} {psa:<6} {delta:<7} {status}")
        print()

    elif args.command == "export-csv":
        p.export_csv(args.output)


if __name__ == "__main__":
    _cli()
