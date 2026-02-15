# PokéGrader

A computer vision tool that estimates Pokémon card condition grades on the PSA 1–10 scale from a phone camera photo.

**Status:** Early development — model training is in progress.

## Idea

Professional card grading (PSA, BGS, CGC) is slow and expensive. The goal here is to give collectors a quick, free estimate by analyzing the same factors the pros look at: centering, corner sharpness, edge wear, and surface condition.

## Approach

- Capture a card image via phone camera or webcam
- Preprocess and segment the card from the background
- Run sub-models for each grading factor (centering, corners, edges, surface)
- Combine into a weighted composite grade (PSA 1–10 scale)

## Tech

Python, FastAPI, PyTorch, OpenCV. Frontend is lightweight HTML/JS using the browser Camera API for mobile capture.

## What's Done / What's Next

- [x] Project architecture
- [ ] Data collection — sourcing card images with known PSA grades
- [ ] Image preprocessing pipeline (crop, normalize, align)
- [ ] Individual factor models (centering, corners, edges, surface)
- [ ] Composite grade prediction
- [ ] Web UI with live camera
- [ ] Confidence scores and grade breakdowns

## Disclaimer

This provides estimated grades for personal reference. Not affiliated with PSA, BGS, CGC, or any official grading service.
