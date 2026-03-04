# PokéGrader

AI-powered Pokémon card condition grading from your phone. Guides you through a multi-view inspection, analyzes centering, corners, edges, and surface quality, then predicts a PSA 1–10 grade.

> **Status:** v2 in development — multi-view guided capture with real-time tilt sensing, interactive AI inspection agent, and per-factor grading models. v1 beta (single-photo centering analysis) is live.

## Try the Beta

**[Launch PokéGrader →](https://huggingface.co/spaces/kevwill/pokegrader)**

Open on your phone, snap a photo of any Pokémon card, get centering analysis with PSA-standard ratios. Works on any mobile browser — no app install needed.

## How It Works

### v2: Multi-View Guided Capture (In Development)

Professional graders don't look at a card from one angle — they tilt it under light, inspect corners up close, and check edges from the side. A single flat photo misses most of what determines a grade. PokéGrader v2 mirrors this process with a guided multi-view capture flow:

**4 standard shots, each with a job:**

| Shot | Angle | What It Captures |
|------|-------|-----------------|
| Front Face | 0° flat | Centering, surface condition, corner crops |
| Back Face | 0° flat, flipped | Back centering, back surface quality |
| Raking Light A | ~30° tilt forward | Top/side edge whitening, surface scratches, holo clouding |
| Raking Light B | ~30° tilt opposite | Bottom/side edge whitening, corner fraying, complementary surface data |

The phone's accelerometer provides a real-time tilt gauge during capture — a ring indicator shows the current angle and lights up green when you're in the target zone (0–10° for flat shots, 25–35° for raking light). The user always knows when they've nailed the angle.

Total capture time: ~15 seconds. Each photo routes to the analysis it's best suited for rather than asking one model to figure out everything from a single frame.

### Interactive AI Inspection Agent (In Development)

After the initial 4 shots, the AI runs a first-pass analysis and identifies regions of interest — potential edge whitening, corner wear, surface scratches — with bounding boxes and confidence scores. If confidence is low on any factor, the AI doesn't guess. It asks.

The flow:

1. **4 guided captures** → front, back, two angled views
2. **First-pass analysis** → centering (algorithmic, high confidence), surface/edge/corner models (variable confidence)
3. **AI flags areas of concern** → highlights specific regions on the card image with what it sees and why it's uncertain
4. **Targeted close-up requests** → "I need a closer look at the top-right edge. Angle the card so light rakes across this area."
5. **User provides close-ups** → AI re-evaluates each flagged region with higher-resolution data
6. **Final grade with reasoning** → full breakdown showing how each factor scored, what was the limiting factor, and why

This creates a collaborative grading experience. The user understands exactly what's affecting the grade, and the AI only makes calls it's confident about.

### v1: What's Live Now

The current beta handles the single-photo flow:

- Card detection and perspective correction from any background using OpenCV
- Algorithmic centering analysis with PSA-standard ratios (e.g., 55/45 left/right)
- Client-side photo quality validation (blur detection via Laplacian variance, exposure checks, glare/hot-spot detection)
- EfficientNet-B0 grading model with metadata fusion (when deployed)
- Mobile-first camera UI on Hugging Face Spaces

## Architecture

### v2: Multi-View Pipeline

```
CAPTURE (4 guided shots with tilt validation)
  │
  ├─ Front Face (0°) ──→ Centering Module (algorithmic, OpenCV)
  │                   ──→ Surface Head (EfficientNet-B0)
  │                   ──→ Corner Crops (4× corner sub-model)
  │
  ├─ Back Face (0°) ───→ Back Centering (algorithmic)
  │                   ──→ Back Surface Head
  │
  ├─ Raking Light A ──→ Edge Head (top + side edge scoring)
  │   (~30° tilt)     ──→ Defect Detector (defect locations + types)
  │                   ──→ Surface Scratch Detection
  │
  └─ Raking Light B ──→ Edge Head (bottom + side edge scoring)
      (~30° opp.)     ──→ Corner Fraying Detection
                      ──→ Surface Diff (cross-view comparison with Shot A)
                          │
FUSION ←──────────────────┘
  │
  ├─ Centering:  20% weight ← Front + Back algorithmic analysis
  ├─ Corners:    25% weight ← Front crops + angled confirmation
  ├─ Edges:      25% weight ← Raking light A + B edge heads
  └─ Surface:    30% weight ← All 4 views fused
  │
OUTPUT
  ├─ Composite grade 1–10 with confidence interval
  ├─ Per-factor sub-scores
  ├─ Identified defects with locations
  └─ Reasoning trail (what's limiting the grade and why)
```

A shared EfficientNet-B0 backbone extracts visual features across all views, with view-specific heads scoring individual grading factors. Each view has a known semantic role, so routing is deterministic — the front face always goes to centering and surface, the angled shots always go to edge detection. Centering stays purely algorithmic because you can calculate it exactly.

### v1: Current Model

The v1 model is a hybrid image + metadata architecture:

- **Image branch:** EfficientNet-B0 → 1280-dim visual features
- **Aux branch:** 11-dim metadata vector (defect counts, centering offsets, defect type flags) → small MLP → 32-dim
- **Fusion:** Concatenated [1312-dim] → FC layers → single continuous grade (1.0–10.0)

Trained for 27 epochs on ~1,200 cards from TAG Grading and eBay scraping:

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 0.94 grades |
| Within 1 grade | 71.7% |
| Within 0.5 grade | 46.6% |
| Exact grade match | 39.3% |

Primary limitation: class imbalance — 86% of training data is grades 8–10, only 14% is grades 1–7.

### Why Multi-View Over Single-Photo

Grading is inherently a 3D assessment compressed into 2D images. A single flat photo loses critical signal:

- **Surface scratches** are invisible unless light hits them at the right angle
- **Edge whitening** is a side-on phenomenon that face-on photos barely capture
- **Holo clouding** only appears under specific lighting conditions
- **Corner wear** can be obscured by photo angle

The raking light shots (3 and 4) are what separate this from every other card grading app. Diffing two angled views can isolate real defects from normal card texture, because true scratches appear in both views while holo refraction shifts.

## Data

### TAG Grading — What's Useful

[TAG Grading](https://my.taggrading.com/) provides scoring data for 591,000+ graded Pokémon cards. Data analysis revealed which signals carry actual training value:

**Useful:** TAG's continuous 0–1000 score (turns classification into regression with gradient signal between similar grades), identified defects with location/category/type and close-up images (free defect-detection annotation), and full card images across the grade spectrum.

**Not useful:** Per-feature sub-scores (Fray, Fill, Angle) for corners and edges show virtually no variance — a grade 1 card scores 994–1000 on corners, same as a grade 10. This finding killed the original plan of training separate per-factor CNNs and drove the pivot to the unified multimodal approach.

### Data Strategy

**Grade diversity over card popularity.** A dinged corner looks the same on a Charizard as a Caterpie. The model learns grading features, not card identity. Biasing toward popular cards risks overfitting to specific layouts.

**Set diversity across eras.** Base Set, modern, Japanese, and other eras have different borders, holo patterns, print quality, and card stock. Training across sets forces the model to generalize the actual grading signals.

**Active class balancing.** ~530 additional low-grade cards (especially grades 4 and 7) being collected to balance the current 86/14 high/low grade skew.

### Training Sources

| Split | Source | Purpose |
|-------|--------|---------|
| Train | eBay PSA listings + TAG reports + phone-camera augmentation | Thousands of labeled images |
| Validate | Original phone photos submitted to PSA | Does the model work on real phone cameras? |
| Test | Held-out phone photos with returned PSA grades | Final accuracy, never seen during training |

As more cards are submitted to PSA, the validation and test sets grow and the model retrains with better real-world signal.

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch, OpenCV
- **Frontend:** HTML/JS, browser Camera API, DeviceOrientationEvent (tilt sensing)
- **Vision Model:** EfficientNet-B0 backbone with view-specific heads (v2) / aux feature MLP (v1)
- **Card Detection:** OpenCV contour detection, perspective correction, portrait normalization
- **Centering:** Algorithmic — Sobel gradient border analysis, PSA-standard ratio calculation
- **Quality Gate:** Client-side blur (Laplacian variance), exposure, and glare detection
- **Augmentation:** Phone camera simulation (blur, lighting, rotation, JPEG artifacts, noise)
- **Deployment:** Docker on Hugging Face Spaces

## Run Locally

```bash
pip install -r requirements.txt
python -m src.api.app
```

Open `http://localhost:8000` on your phone (same Wi-Fi network).

## Project Structure

```
pokegrader/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI server + grading endpoints
│   ├── model/
│   │   ├── grader.py           # EfficientNet-B0 + aux feature fusion model
│   │   ├── train.py            # Training loop (eBay data)
│   │   └── train_tag.py        # Training loop (TAG data)
│   └── utils/
│       ├── card_detector.py    # Card detection & perspective correction
│       ├── centering.py        # Centering analysis (algorithmic)
│       ├── preprocessing.py    # Image preprocessing & quality validation
│       └── augmentation.py     # Phone camera augmentation
├── configs/
│   └── default.yaml            # Model & grading config
├── data/
│   ├── raw/                    # Scraped training images
│   ├── processed/              # Cropped & organized
│   └── models/                 # Trained model checkpoints
├── web/
│   └── templates/
│       └── index.html          # Multi-view capture UI
├── Dockerfile                  # HF Spaces deployment
└── README.md
```

## Grading Factors

| Factor | Weight | Method | Status |
|--------|--------|--------|--------|
| Centering | 20% | Algorithmic (OpenCV border analysis) | ✅ Live |
| Corners | 25% | View-specific head (front crops + angled confirmation) | 🔧 Building |
| Edges | 25% | View-specific head (raking light A + B) | 🔧 Building |
| Surface | 30% | Multi-view fusion (all 4 shots) | 🔧 Building |

## Roadmap

- [x] Card detection and perspective correction
- [x] Algorithmic centering analysis (PSA-standard ratios)
- [x] Client-side photo quality validation
- [x] EfficientNet-B0 + metadata fusion model (v1)
- [x] Mobile-first camera UI deployed on HF Spaces
- [ ] Multi-view guided capture with tilt sensing
- [ ] Interactive AI inspection agent (ROI flagging + targeted close-ups)
- [ ] Per-factor grading heads (corners, edges, surface)
- [ ] Balanced dataset retraining (~530 low-grade cards)
- [ ] Multi-view backend endpoint (`/api/grade-multiview`)
- [ ] Cross-view surface diffing (raking light comparison)
- [ ] Defect detection with bounding boxes

## Disclaimer

Estimated grades for personal reference only. Not affiliated with PSA, BGS, CGC, or any official grading service.
