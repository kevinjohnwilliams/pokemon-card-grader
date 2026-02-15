# 🃏 PokéGrader — AI-Powered Pokémon Card Condition Grader

**Instantly grade your Pokémon cards using your phone's camera.**

PokéGrader uses computer vision and deep learning to analyze Pokémon trading cards and assign a condition grade on the **PSA 1–10 scale** — no submission fees, no waiting weeks.

> ⚠️ **Status: Early Development** — Model training is in progress. Star/watch this repo to follow along.

---

## 🎯 What It Does

1. **Snap a photo** of your Pokémon card using your phone or webcam
2. **AI analyzes** the card for surface scratches, edge wear, corner damage, and centering
3. **Get an estimated PSA grade** (1–10) with a breakdown of each grading factor

## 📊 Grading Criteria

PokéGrader evaluates the same factors professional graders use:

| Factor       | Description                                      | Weight |
|--------------|--------------------------------------------------|--------|
| **Centering** | Border symmetry on front and back                | 20%    |
| **Corners**   | Sharpness and wear on all four corners           | 25%    |
| **Edges**     | Chipping, nicks, and wear along card edges       | 25%    |
| **Surface**   | Scratches, print defects, whitening, holo damage | 30%    |

## 🏗️ Project Structure

```
pokemon-card-grader/
├── src/
│   ├── model/          # Model architecture, training, and inference
│   ├── api/            # FastAPI backend for serving predictions
│   └── utils/          # Image preprocessing, grading logic
├── web/
│   ├── templates/      # HTML templates (camera UI)
│   └── static/         # CSS, JS (camera capture, results display)
├── data/
│   ├── raw/            # Original card images
│   ├── processed/      # Cleaned and normalized images
│   └── augmented/      # Augmented training data
├── notebooks/          # Exploration and model experimentation
├── configs/            # Model and app configuration
├── tests/              # Unit and integration tests
└── docs/               # Additional documentation
```

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI
- **ML/CV:** PyTorch, torchvision, OpenCV
- **Frontend:** HTML/CSS/JS with browser Camera API (mobile-friendly)
- **Training:** Custom CNN / fine-tuned EfficientNet

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/pokemon-card-grader.git
cd pokemon-card-grader
pip install -r requirements.txt
```

### Run the App

```bash
python -m src.api.app
```

Then open `http://localhost:8000` on your phone or desktop browser.

## 🗺️ Roadmap

- [x] Project structure and architecture
- [ ] Data collection pipeline (card images with known PSA grades)
- [ ] Image preprocessing (crop, normalize, alignment)
- [ ] Sub-model training (centering, corners, edges, surface)
- [ ] Composite grade prediction (weighted ensemble → PSA 1–10)
- [ ] Web UI with live camera capture
- [ ] Confidence score and grade explanation
- [ ] Mobile PWA support
- [ ] Batch grading (multiple cards)
- [ ] Price estimation based on grade + card ID

## 🤝 Contributing

This project is in early development. If you're interested in contributing — especially with labeled card image datasets — please open an issue or reach out!

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## ⚖️ Disclaimer

PokéGrader provides **estimated grades for personal reference only**. It is not affiliated with PSA, BGS, CGC, or any official grading service. Grades are approximations and should not be used as a substitute for professional grading.
