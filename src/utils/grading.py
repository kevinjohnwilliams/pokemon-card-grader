"""
PokéGrader — Grading Logic

Combines sub-factor scores into a composite PSA-style grade (1-10).
"""

from dataclasses import dataclass

# Default weights matching PSA grading emphasis
DEFAULT_WEIGHTS = {
    "centering": 0.20,
    "corners": 0.25,
    "edges": 0.25,
    "surface": 0.30,
}

# Composite score thresholds → PSA grade
GRADE_THRESHOLDS = [
    (0.95, 10, "Gem Mint"),
    (0.85, 9, "Mint"),
    (0.75, 8, "Near Mint-Mint"),
    (0.65, 7, "Near Mint"),
    (0.55, 6, "Excellent-Mint"),
    (0.45, 5, "Excellent"),
    (0.35, 4, "Very Good-Excellent"),
    (0.25, 3, "Very Good"),
    (0.15, 2, "Good"),
    (0.00, 1, "Poor"),
]


@dataclass
class GradeResult:
    """Complete grading result with sub-scores and final grade."""
    grade: int
    grade_label: str
    composite_score: float
    centering: float
    corners: float
    edges: float
    surface: float
    confidence: float


def compute_grade(
    centering: float,
    corners: float,
    edges: float,
    surface: float,
    weights: dict[str, float] | None = None,
) -> GradeResult:
    """
    Compute a PSA-style grade from sub-factor scores.
    
    Args:
        centering: Centering quality score (0-1)
        corners: Corner condition score (0-1)
        edges: Edge condition score (0-1)
        surface: Surface condition score (0-1)
        weights: Optional custom weights for each factor
    
    Returns:
        GradeResult with final grade and breakdown
    """
    w = weights or DEFAULT_WEIGHTS
    
    composite = (
        centering * w["centering"]
        + corners * w["corners"]
        + edges * w["edges"]
        + surface * w["surface"]
    )

    grade = 1
    label = "Poor"
    for threshold, g, l in GRADE_THRESHOLDS:
        if composite >= threshold:
            grade = g
            label = l
            break

    return GradeResult(
        grade=grade,
        grade_label=label,
        composite_score=round(composite, 4),
        centering=round(centering, 4),
        corners=round(corners, 4),
        edges=round(edges, 4),
        surface=round(surface, 4),
        confidence=0.0,  # TODO: Derive from model output
    )
