# PokéGrader

AI-powered Pokémon card condition grading from a phone camera photo. Estimates PSA 1–10 grades by analyzing centering, corners, edges, and surface quality.

> **Status:** Working prototype — card detection and centering analysis are live. Multimodal grading pipeline in development.

## How It Works

Snap a photo of a card with your phone → the app detects the card, crops it, and predicts a PSA-style grade based on the same factors professional graders evaluate.

**What's working now:**
- Card detection and perspective correction from any background
- Real centering analysis with PSA-standard ratios (e.g., 55/45 left/right)
- Mobile-first camera UI with photo quality validation (blur, exposure, glare detection)
- Training data collection pipeline (~1,200+ labeled cards from TAG Grading)

**What's next:**
- Multimodal grading pipeline (vision encoder → agent → composite grade)
- Defect and ding detection from full card images
- Composite grade prediction with per-factor breakdowns and confidence scores

## Architecture: Multimodal Grading Pipeline

PokéGrader uses a multimodal pipeline that mirrors how a human grader evaluates a card — combining visual analysis with structured measurements rather than relying on a single end-to-end model.

```
┌─────────────────────────────────────────────────────────┐
│                     QUERY (Input)                       │
│  Raw photo → card detection → crop & perspective fix    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   ENCODER (Features)                    │
│  Pretrained vision encoder (ViT / CLIP) produces rich   │
│  feature embeddings from the cropped card image         │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   AGENT (Reasoning)                     │
│  Orchestrates specialized analyses:                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Centering   │  │   Defect /   │  │   Surface    │  │
│  │ (algorithmic)│  │ Ding Detect  │  │  Condition   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│  Fuses visual features + algorithmic results + metadata │
│  (defect counts, centering ratios, wear indicators)     │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   OUTPUT (Grade)                        │
│  PSA 1–10 grade + per-factor breakdown + confidence     │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture

Early data analysis revealed that TAG Grading's per-feature sub-scores (Fray, Fill, Angle for each corner and edge) have virtually no variance across grade levels — a grade 1 card and a grade 10 card both score 994–1000 on corners and edges. This made the original plan of training separate CNN sub-models per grading factor unviable.

What *does* carry strong signal: the overall TAG score (0–1000), defect counts and types, centering measurements, and the visual appearance of the full card image. The multimodal pipeline exploits all of these by combining a powerful vision encoder with structured data fusion, rather than asking a single model to learn everything from pixels alone.

### Pipeline Stages

**Query (Preprocessing)** — The raw phone photo is validated for quality (blur, exposure, glare), then the card is detected, perspective-corrected, and cropped. This gate ensures only usable images reach the model.

**Encoder (Feature Extraction)** — A pretrained vision encoder (ViT or CLIP) produces dense feature embeddings from the cropped card. Using a frozen pretrained encoder lets us leverage representations trained on millions of images without needing a massive card-specific dataset.

**Agent (Analysis & Fusion)** — The orchestration layer routes encoded features through specialized checks: algorithmic centering analysis (already working), defect/ding detection, and overall condition assessment. It then fuses visual features with structured metadata (defect counts, centering ratios) through a learned fusion head to produce the final grade. This mirrors a human grader's process of evaluating multiple factors and synthesizing a judgment.

**Output** — A composite PSA-style grade (1–10) with per-factor breakdowns, confidence scores, and interpretable reasoning about what drove the grade.

## Training Strategy

The model trains on two data sources: scraped eBay images of PSA-graded cards and TAG Grading's scoring reports. Validation and testing use our own phone photos submitted to PSA. Augmentation simulates real phone camera conditions (blur, lighting, noise) on the training data to close the gap between clean listing photos and real-world usage.

<p align="center">
  <img src="docs/pipeline-diagram.svg" alt="PokéGrader Training Framework" width="800"/>
</p>

| Split | Source | Purpose |
|-------|--------|---------|
| **Train** | Scraped eBay PSA listings + TAG reports + augmentation | Thousands of images with rich labels |
| **Validate** | Our phone photos → submitted to PSA | Reality check — does the model work on real phone cameras? |
| **Test** | Held-out phone photos with PSA grades | Final accuracy measurement, never seen during training |

The feedback loop: as we submit more cards to PSA, the validation and test sets grow, and the model gets retrained with better real-world signal.

## Data Collection

### TAG Grading — What's Useful (and What Isn't)

[TAG Grading](https://my.taggrading.com/) provides detailed scoring data for over 591,000 graded Pokémon cards. Through data analysis, we identified which signals are actually useful for training:

**✅ TAG Score (0–1000)** — A continuous score mapping to the final 1–10 grade. Turns classification into regression, giving gradient signal even between similar cards (e.g., two "10 GEM MINT" cards scoring 985 vs 970).

**✅ Identified Defects** — Specific defect instances with location, category (SURFACE, EDGE, CORNER), type (INK DEFECT, etc.), and close-up images. Defect counts correlate strongly with grade — grade 10s have 0–1 defects, grade 6s have 4+. This is essentially free defect-detection annotation.

**✅ Full Card Images** — 1,200+ card images across the grade spectrum with known grades, usable as direct training data for the vision encoder.

**❌ Per-Feature Sub-Scores** — Fray, Fill, and Angle scores (each 0–1000) for corners and edges show virtually no variance across grades. A grade 1 card scores 994–1000 on corners, same as a grade 10. Many cards have these fields completely empty. These sub-scores are **not usable** as independent training labels.

This finding drove the architectural pivot from separate per-factor CNNs to the unified multimodal pipeline described above.

### Data Strategy

**Prioritize grade diversity, not card popularity.** A dinged corner looks the same on a Charizard as it does on a Caterpie. The model needs to learn grading features (edge whitening, corner fraying, surface scratches), not which Pokémon is on the card. Biasing toward popular cards risks overfitting to specific layouts and color patterns.

**Prioritize set diversity.** Different eras have different border styles, holo patterns, print quality, and card stock. Training across multiple sets (Base Set, modern, Japanese, etc.) forces the model to generalize the actual grading signals rather than memorizing set-specific visual patterns.

**Address class imbalance.** The current dataset skews heavily toward grades 8–10. Active collection of ~530 additional low-grade cards (especially grades 4 and 7) is in progress to balance the training distribution.

| Priority | What | Why |
|----------|------|-----|
| 🔴 High | Grade distribution (spread across 1–10 and 0–1000) | Model needs examples of every condition level |
| 🔴 High | Set/era diversity (Base Set, modern, Japanese, etc.) | Generalize across border styles and print quality |
| 🔴 High | Class balance (low-grade card collection) | Prevent model from defaulting to high grades |
| 🟡 Medium | Defect type coverage (ink, surface, corner wear, etc.) | Defect detection needs variety |
| 🟢 Low | Card popularity (Charizard vs Caterpie) | Grading features are card-agnostic |

## Tech Stack

- **Backend:** Python, FastAPI, OpenCV, PyTorch
- **Frontend:** HTML/JS with browser Camera API (mobile-first)
- **CV Pipeline:** Card detection, perspective correction, border analysis, photo quality validation
- **Vision Encoder:** Pretrained ViT/CLIP (frozen) for feature extraction
- **Fusion Head:** Lightweight MLP combining visual features + structured metadata
- **Centering:** Algorithmic (OpenCV border analysis) — no model needed
- **Augmentation:** Phone camera simulation (blur, lighting, rotation, JPEG artifacts, noise)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m src.api.app
```

Open `http://localhost:8000` on your phone (same network) to start grading.

## Project Structure

```
pokegrader/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI server
│   ├── model/
│   │   ├── grader.py           # Vision encoder + fusion head
│   │   └── train.py            # Training loop
│   └── utils/
│       ├── card_detector.py    # Card detection & perspective correction
│       ├── centering.py        # Centering analysis (algorithmic)
│       ├── grading.py          # Composite grade calculation
│       ├── preprocessing.py    # Image preprocessing & quality validation
│       ├── pipeline.py         # Data pipeline (capture → PSA → train)
│       └── augmentation.py     # Phone camera augmentation
├── configs/
│   └── default.yaml            # Model & grading config
├── data/
│   ├── raw/                    # Scraped training images
│   ├── processed/              # Cropped & organized
│   └── augmented/              # Augmented training sets
├── web/
│   ├── static/
│   └── templates/
└── tests/
```

## Grading Factors

| Factor | Method | Signal Source | Status |
|--------|--------|--------------|--------|
| Centering | Algorithmic (OpenCV border analysis) | TAG centering measurements | ✅ Working |
| Corners | Vision encoder + fusion head | Full card images + defect metadata | 🔄 Training |
| Edges | Vision encoder + fusion head | Full card images + defect metadata | 🔄 Training |
| Surface | Vision encoder + fusion head | Full card images + defect annotations | 🔄 Training |
| **Composite** | **Agent fusion (visual + structured)** | **All factors combined** | **🔄 Training** |

> **Note:** Corners, edges, and surface are evaluated holistically by the vision encoder rather than through separate per-factor models, since TAG's per-feature sub-scores lack the variance needed for independent training.

## Disclaimer

Estimated grades for personal reference only. Not affiliated with PSA, BGS, CGC, or any official grading service.