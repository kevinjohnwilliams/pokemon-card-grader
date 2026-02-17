# PokéGrader

AI-powered Pokémon card condition grading from a phone camera photo. Estimates PSA 1–10 grades by analyzing centering, corners, edges, and surface quality.

> **Status:** Working prototype — card detection and centering analysis are live. Model training framework is built, data collection in progress.

## How It Works

Snap a photo of a card with your phone → the app detects the card, crops it, and predicts a PSA-style grade based on the same factors professional graders evaluate.

**What's working now:**
- Card detection and perspective correction from any background
- Real centering analysis with PSA-standard ratios (e.g., 55/45 left/right)
- Mobile-first camera UI
- Training pipeline and data augmentation framework

**What's next:**
- Model training once scrape data is ready
- Corner, edge, and surface condition analysis
- Composite grade prediction with confidence scores

## Training Strategy

The model trains on scraped eBay images of PSA-graded cards, validates against our own phone photos with known PSA grades, and is tested on a held-out set of those same phone photos. Augmentation simulates real phone camera conditions (blur, lighting, noise) on the training data to close the gap between clean listing photos and real-world usage.

<p align="center">
  <img src="docs/pipeline-diagram.svg" alt="PokéGrader Training Framework" width="800"/>
</p>

| Split | Source | Purpose |
|-------|--------|---------|
| **Train** | Scraped eBay PSA listings + augmentation | Thousands of images to learn grading patterns fast |
| **Validate** | Our phone photos → submitted to PSA | Reality check — does the model work on real phone cameras? |
| **Test** | Held-out phone photos with PSA grades | Final accuracy measurement, never seen during training |

The feedback loop: as we submit more cards to PSA, the validation and test sets grow, and the model gets retrained with better real-world signal.

## Tech Stack

- **Backend:** Python, FastAPI, OpenCV, PyTorch
- **Frontend:** HTML/JS with browser Camera API (mobile-first)
- **CV Pipeline:** Card detection, perspective correction, border analysis
- **Model:** EfficientNet-B0 fine-tuned on PSA-graded card images
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
│   │   ├── grader.py           # EfficientNet multi-head model
│   │   └── train.py            # Training loop
│   └── utils/
│       ├── card_detector.py    # Card detection & perspective correction
│       ├── centering.py        # Centering analysis (algorithmic)
│       ├── grading.py          # Composite grade calculation
│       ├── preprocessing.py    # Image preprocessing pipeline
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

| Factor | Weight | Method | Status |
|--------|--------|--------|--------|
| Centering | 20% | Algorithmic (border analysis) | ✅ Working |
| Corners | 25% | CNN (fine-tuned EfficientNet) | 🔄 Training data in progress |
| Edges | 25% | CNN (fine-tuned EfficientNet) | 🔄 Training data in progress |
| Surface | 30% | CNN (fine-tuned EfficientNet) | 🔄 Training data in progress |

## Disclaimer

Estimated grades for personal reference only. Not affiliated with PSA, BGS, CGC, or any official grading service.