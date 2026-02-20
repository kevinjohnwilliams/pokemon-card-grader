"""
PokéGrader — TAG Data Audit

Quick audit of scraped TAG data to catch issues before training.

Checks for:
- Missing/corrupt images
- Slab images incorrectly saved as edge crops
- Grade distribution (need spread across 1-10)
- Missing scores (fray/fill/angle)
- Inconsistent metadata

Usage:
    python -m scripts.audit_tag_data tag_dataset/cards
    python -m scripts.audit_tag_data tag_dataset/cards --check-images
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def audit(cards_dir: str, check_images: bool = False):
    """Run full audit on TAG dataset."""
    cards_path = Path(cards_dir)

    if not cards_path.exists():
        print(f"❌ Directory not found: {cards_dir}")
        sys.exit(1)

    meta_files = sorted(cards_path.glob("*/*_metadata.json"))
    print(f"Found {len(meta_files)} card folders\n")

    # Counters
    total_cards = 0
    grade_dist = Counter()
    score_ranges = {"tag_score": [], "fray": [], "fill": [], "angle": []}
    issues = defaultdict(list)

    corner_count = 0
    edge_count = 0
    defect_count = 0
    missing_angle = 0
    slab_as_edge = 0
    missing_images = 0

    for meta_file in meta_files:
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            issues["corrupt_json"].append(f"{meta_file}: {e}")
            continue

        cert_id = meta.get("cert_id", meta_file.parent.name)
        card_dir = meta_file.parent
        total_cards += 1

        # ---- Grade validation ----
        grade = meta.get("grade", {}).get("number")
        if grade is None:
            issues["missing_grade"].append(cert_id)
            continue
        if not (1 <= grade <= 10):
            issues["invalid_grade"].append(f"{cert_id}: grade={grade}")
            continue
        grade_dist[int(grade)] += 1

        tag_score = meta.get("tag_score")
        if tag_score is not None:
            score_ranges["tag_score"].append(tag_score)

        # ---- Corner audit ----
        corners_meta = meta.get("corners", {})
        corners_images = meta.get("images", {}).get("corners", {})

        for corner_key, corner_data in corners_meta.items():
            for side in ("front", "back"):
                side_data = corner_data.get(side, {})
                if side_data:
                    corner_count += 1
                    if "fray" in side_data:
                        score_ranges["fray"].append(side_data["fray"])
                    if "fill" in side_data:
                        score_ranges["fill"].append(side_data["fill"])
                    if "angle" in side_data:
                        score_ranges["angle"].append(side_data["angle"])
                    else:
                        missing_angle += 1

        # ---- Edge audit ----
        edges_meta = meta.get("edges", {})
        edges_images = meta.get("images", {}).get("edges", {})

        for side in ("front", "back"):
            for edge_img in edges_images.get(side, []):
                edge_count += 1
                url = edge_img.get("url", "")
                if "Slabbed" in url:
                    slab_as_edge += 1
                    issues["slab_as_edge"].append(
                        f"{cert_id}: {edge_img['position']} {side} → {url.split('/')[-1]}"
                    )

        # ---- Defect audit ----
        defects = meta.get("defects", [])
        defect_count += len(defects)

        # ---- Image existence check ----
        if check_images:
            images = meta.get("images", {})
            for key in ("front", "back"):
                rel = images.get(key)
                if rel:
                    fname = Path(rel).name
                    if not (card_dir / fname).exists():
                        missing_images += 1
                        issues["missing_images"].append(f"{cert_id}: {key}")

            for side in ("front", "back"):
                for corner in corners_images.get(side, []):
                    fname = Path(corner["path"]).name
                    if not (card_dir / "corners" / fname).exists():
                        missing_images += 1
                        issues["missing_images"].append(
                            f"{cert_id}: corner {corner['position']} {side}"
                        )

                for edge in edges_images.get(side, []):
                    fname = Path(edge["path"]).name
                    if not (card_dir / "edges" / fname).exists():
                        missing_images += 1
                        issues["missing_images"].append(
                            f"{cert_id}: edge {edge['position']} {side}"
                        )

    # ---- Print report ----
    print("=" * 60)
    print("TAG DATA AUDIT REPORT")
    print("=" * 60)

    print(f"\n📊 OVERVIEW")
    print(f"  Total cards:      {total_cards}")
    print(f"  Corner samples:   {corner_count}")
    print(f"  Edge samples:     {edge_count}")
    print(f"  Defect records:   {defect_count}")

    print(f"\n📈 GRADE DISTRIBUTION")
    for grade in range(1, 11):
        count = grade_dist.get(grade, 0)
        bar = "█" * (count // 2) if count > 0 else ""
        print(f"  Grade {grade:2d}: {count:4d} {bar}")
    print(f"  Total:   {sum(grade_dist.values()):4d}")

    print(f"\n📐 SCORE RANGES (0-1000)")
    for name, values in score_ranges.items():
        if values:
            print(f"  {name:10s}: min={min(values):4.0f}  max={max(values):4.0f}  "
                  f"mean={sum(values)/len(values):6.1f}  n={len(values)}")

    print(f"\n⚠️  KNOWN ISSUES")
    print(f"  Missing angle scores: {missing_angle} (normal — not all have angle)")
    print(f"  Slab images as edges: {slab_as_edge} (these should be skipped in training)")
    if check_images:
        print(f"  Missing image files:  {missing_images}")

    if any(issues[k] for k in ("corrupt_json", "missing_grade", "invalid_grade")):
        print(f"\n❌ DATA ERRORS")
        for issue_type in ("corrupt_json", "missing_grade", "invalid_grade"):
            if issues[issue_type]:
                print(f"  {issue_type}: {len(issues[issue_type])}")
                for item in issues[issue_type][:5]:
                    print(f"    - {item}")
                if len(issues[issue_type]) > 5:
                    print(f"    ... and {len(issues[issue_type]) - 5} more")

    # ---- Training readiness ----
    print(f"\n{'=' * 60}")
    print("TRAINING READINESS")
    print("=" * 60)

    ready = True

    # Check grade spread
    grades_present = len([g for g in range(1, 11) if grade_dist.get(g, 0) > 0])
    if grades_present < 5:
        print(f"  ⚠️  Only {grades_present}/10 grades represented — model may struggle")
        ready = False
    else:
        print(f"  ✅ {grades_present}/10 grades represented")

    # Check minimum samples
    if total_cards < 100:
        print(f"  ⚠️  Only {total_cards} cards — consider collecting more")
    else:
        print(f"  ✅ {total_cards} cards (target: 500+)")

    # Check corner data
    expected_corners = total_cards * 8
    if corner_count >= expected_corners * 0.8:
        print(f"  ✅ {corner_count} corner samples (~{corner_count // total_cards}/card)")
    else:
        print(f"  ⚠️  {corner_count} corner samples (expected ~{expected_corners})")

    # Slab image warning
    usable_edges = edge_count - slab_as_edge
    print(f"  {'✅' if slab_as_edge == 0 else '⚠️'}  {usable_edges} usable edge samples "
          f"({slab_as_edge} slab images filtered)")

    if ready:
        print(f"\n  🚀 Data looks ready for training!")
        print(f"\n  Run:")
        print(f"    python -m src.model.train_tag --model corners --data-dir {cards_dir}")
        print(f"    python -m src.model.train_tag --model edges --data-dir {cards_dir}")
        print(f"    python -m src.model.train_tag --model overall --data-dir {cards_dir}")

    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit TAG dataset")
    parser.add_argument("cards_dir", help="Path to TAG cards directory")
    parser.add_argument("--check-images", action="store_true",
                        help="Also verify all image files exist (slower)")
    args = parser.parse_args()

    audit(args.cards_dir, check_images=args.check_images)