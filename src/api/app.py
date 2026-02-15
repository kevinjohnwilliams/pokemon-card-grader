"""
PokéGrader — API Server

FastAPI app that serves card detection and centering analysis.
Accepts card images via upload or camera capture.

Usage:
    python3 -m src.api.app
"""

import io
import base64
import numpy as np
import cv2
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils.card_detector import detect_and_crop_card, ensure_portrait
from src.utils.centering import analyze_centering

app = FastAPI(
    title="PokéGrader",
    description="AI-powered Pokémon card condition grading",
    version="0.2.0",
)

# Allow CORS for local dev
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


@app.get("/")
async def home():
    """Serve the camera capture UI."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "PokéGrader API is running. Upload a card image to /api/grade"}


@app.post("/api/grade")
async def grade_card(file: UploadFile = File(...)):
    """
    Grade a Pokémon card from an uploaded image.

    Currently performs:
    - Card detection and cropping
    - Centering analysis (real, algorithmic)
    - Placeholder scores for corners, edges, surface

    Returns:
        JSON with grade breakdown and card image data.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Step 1: Detect and crop card
    cropped, debug_img, detection_info = detect_and_crop_card(image, debug=True)

    if cropped is None:
        return JSONResponse(content={
            "success": False,
            "error": "no_card_detected",
            "message": "Could not detect a card in the image. Try placing the card on a contrasting background with good lighting.",
        })

    # Ensure portrait orientation
    cropped = ensure_portrait(cropped)

    # Step 2: Analyze centering (REAL analysis)
    centering = analyze_centering(cropped)

    # Step 3: Placeholder scores for other factors (until ML model is ready)
    corners_score = None
    edges_score = None
    surface_score = None

    # Step 4: Compute partial grade (only centering is real)
    partial_grade = centering.grade_estimate

    # Encode cropped card as base64 for the frontend
    _, buffer = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cropped_b64 = base64.b64encode(buffer).decode('utf-8')

    # Encode debug image if available
    debug_b64 = None
    if debug_img is not None:
        _, dbuf = cv2.imencode('.jpg', debug_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        debug_b64 = base64.b64encode(dbuf).decode('utf-8')

    return JSONResponse(content={
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
        "corners": {
            "score": corners_score,
            "status": "coming_soon",
        },
        "edges": {
            "score": edges_score,
            "status": "coming_soon",
        },
        "surface": {
            "score": surface_score,
            "status": "coming_soon",
        },
        "grade": {
            "value": partial_grade,
            "is_partial": True,
            "note": "Grade based on centering only. Corners, edges, and surface analysis coming soon.",
        },
        "images": {
            "cropped_card": cropped_b64,
            "debug_detection": debug_b64,
        },
    })


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "features": {
            "card_detection": True,
            "centering_analysis": True,
            "corners_analysis": False,
            "edges_analysis": False,
            "surface_analysis": False,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
