"""
PokéGrader — API Server

FastAPI app that serves card detection, centering analysis, and
model-based grade prediction when a trained model is available.

Usage:
    python -m src.api.app
    python -m src.api.app --model data/models/run_20250615/best_model.pt
"""

import base64
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils.card_detector import detect_and_crop_card, ensure_portrait
from src.utils.centering import analyze_centering

app = FastAPI(
    title="PokéGrader",
    description="AI-powered Pokémon card condition grading",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "static"
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "web" / "templates"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_model_device = None
_model_transform = None
_model_info = {}


def load_model(checkpoint_path: Optional[str] = None):
    """
    Load trained model checkpoint. Called at startup.
    Auto-discovers the latest model if no path is given.
    """
    global _model, _model_device, _model_transform, _model_info

    # Auto-discover latest model
    if checkpoint_path is None:
        models_dir = Path(__file__).parent.parent.parent / "data" / "models"
        if models_dir.exists():
            runs = sorted(models_dir.glob("run_*/best_model.pt"))
            if runs:
                checkpoint_path = str(runs[-1])

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("  ℹ️  No trained model found — centering-only grading")
        return

    try:
        from src.model.grader import CardGraderModel
        from torchvision import transforms

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = CardGraderModel(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        _model = model
        _model_device = device
        _model_transform = transform
        _model_info = {
            "path": checkpoint_path,
            "epoch": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
        }

        print(f"  ✅ Model loaded: {checkpoint_path}")
        print(f"     Epoch {_model_info['epoch']}, "
              f"val acc: {_model_info['val_accuracy']:.1f}%")

    except Exception as e:
        print(f"  ⚠️  Failed to load model: {e}")


def predict_with_model(card_image: np.ndarray) -> Optional[dict]:
    """
    Run the trained model on a cropped card image.
    Returns None if no model is loaded.
    """
    if _model is None:
        return None

    try:
        rgb = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
        tensor = _model_transform(rgb).unsqueeze(0).to(_model_device)

        with torch.no_grad():
            logits = _model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        confidence, pred_idx = probs.max(dim=0)
        grade = pred_idx.item() + 1

        grade_probs = {
            str(i + 1): round(probs[i].item(), 4)
            for i in range(10)
        }

        return {
            "grade": grade,
            "confidence": round(confidence.item(), 4),
            "probabilities": grade_probs,
        }
    except Exception as e:
        print(f"  ⚠️  Model prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    load_model()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def home():
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "PokéGrader API is running. POST to /api/grade"}


@app.post("/api/grade")
async def grade_card(file: UploadFile = File(...)):
    """
    Grade a Pokémon card from an uploaded image.

    Returns grade breakdown with centering analysis and model prediction
    (when a trained model is available).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Detect and crop
    cropped, debug_img, detection_info = detect_and_crop_card(image, debug=True)

    if cropped is None:
        return JSONResponse(content={
            "success": False,
            "error": "no_card_detected",
            "message": "Could not detect a card. "
                       "Try a contrasting background with good lighting.",
        })

    cropped = ensure_portrait(cropped)

    # Centering (always runs — algorithmic, no model needed)
    centering = analyze_centering(cropped)

    # Model prediction (if available)
    model_prediction = predict_with_model(cropped)

    # Determine final grade
    if model_prediction:
        final_grade = model_prediction["grade"]
        grade_source = "model"
        is_partial = False
        grade_note = (
            f"AI model prediction "
            f"(confidence: {model_prediction['confidence']:.0%}). "
            f"Centering analyzed algorithmically."
        )
    else:
        final_grade = centering.grade_estimate
        grade_source = "centering_only"
        is_partial = True
        grade_note = (
            "Grade based on centering only. "
            "Full AI grading available when model is trained."
        )

    # Encode images for frontend
    _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cropped_b64 = base64.b64encode(buffer).decode('utf-8')

    debug_b64 = None
    if debug_img is not None:
        _, dbuf = cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        debug_b64 = base64.b64encode(dbuf).decode('utf-8')

    response = {
        "success": True,
        "detection": {
            "detected": detection_info["detected"],
            "confidence": round(detection_info["confidence"], 3),
        },
        "centering": {
            "score": centering.score,
            "grade_estimate": centering.grade_estimate,
            "left_right_ratio": centering.left_right_ratio,
            "top_bottom_ratio": centering.top_bottom_ratio,
            "borders": {
                "left": centering.left_border,
                "right": centering.right_border,
                "top": centering.top_border,
                "bottom": centering.bottom_border,
            },
            "horizontal_offset_pct": centering.horizontal_offset_pct,
            "vertical_offset_pct": centering.vertical_offset_pct,
            "details": centering.details,
        },
        "grade": {
            "value": final_grade,
            "source": grade_source,
            "is_partial": is_partial,
            "note": grade_note,
        },
        "model": (
            {
                "grade": model_prediction["grade"],
                "confidence": model_prediction["confidence"],
                "probabilities": model_prediction["probabilities"],
            }
            if model_prediction
            else {"status": "not_loaded"}
        ),
        "corners": {"score": None, "status": "coming_soon"},
        "edges": {"score": None, "status": "coming_soon"},
        "surface": {"score": None, "status": "coming_soon"},
        "images": {
            "cropped_card": cropped_b64,
            "debug_detection": debug_b64,
        },
    }

    return JSONResponse(content=response)


@app.get("/api/health")
async def health():
    model_loaded = _model is not None
    return {
        "status": "ok",
        "version": "0.3.0",
        "model": {
            "loaded": model_loaded,
            "info": _model_info if model_loaded else None,
        },
        "features": {
            "card_detection": True,
            "centering_analysis": True,
            "model_grading": model_loaded,
            "corners_analysis": False,
            "edges_analysis": False,
            "surface_analysis": False,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)