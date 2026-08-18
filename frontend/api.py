"""
Drift-Sense hackathon demo backend.

Thin HTTP wrapper around the EXISTING, unmodified inference pipeline in
localize.py / matching.py. No prediction logic lives in this file -- it
only handles image upload, calls localize_cv_only() (the same
verified-best, no-CNN mode localize.py uses by default), and formats the
response for the frontend.

GROUND TRUTH -- where it actually comes from:
Ground truth only exists because generate_dataset.py baked it into
metadata.json / ground_truth.csv at the moment it created each image
pair. For an arbitrary uploaded image there is no metadata to read --
nothing in the project can compute a "true" location for a truly new
image, and this app does not invent one.

What it CAN do, honestly: if the uploaded reference/search images are
byte-for-byte the same files generate_dataset.py already produced (e.g.
the presenter re-uploads sample_data/sample_0000/reference.png and
search.png), the backend recognizes them by content hash and looks up
their real GT_X/GT_Y straight out of that sample's existing
ground_truth.csv row -- no manual entry, no fabrication. This is the
"smallest necessary backend modification" to expose metadata that
already exists in the project. If no match is found, ground truth is
simply omitted (has_gt: false) rather than guessed.

At startup this file scans REPO_ROOT for any subdirectory containing a
ground_truth.csv (i.e. any dataset generate_dataset.py produced) and
indexes it. Point DRIFT_SENSE_DATASET_DIRS at extra folders (os.pathsep-
separated) if your generated dataset lives outside the repo.

Run from the webapp/ directory:
    uvicorn api:app --reload --port 8000
"""

import hashlib
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# localize.py / matching.py live in the repo root, one level up from webapp/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from localize import localize_cv_only  # noqa: E402  (existing, unmodified pipeline)

app = FastAPI(title="Drift-Sense Demo API")

# Same-origin in normal use (frontend is served by this app below), but
# CORS is left open so the page can also be opened from a different
# origin/port during development without extra config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_upload_to_gray(raw_bytes: bytes):
    """Mirrors localize.py's load_gray(), but from in-memory bytes
    instead of a file path (images arrive as uploads, not paths)."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img  # None if unreadable/corrupt/unsupported format


def discover_dataset_dirs(root):
    """Any subdirectory that contains a ground_truth.csv is a dataset
    generate_dataset.py produced. Auto-discovering means no hardcoded
    folder name (sample_data, synthetic_sem_dataset, ...) is assumed."""
    found = []
    if not os.path.isdir(root):
        return found
    for entry in os.listdir(root):
        candidate = os.path.join(root, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "ground_truth.csv")):
            found.append(candidate)
    extra = os.environ.get("DRIFT_SENSE_DATASET_DIRS", "")
    for p in extra.split(os.pathsep):
        p = p.strip()
        if p and os.path.isfile(os.path.join(p, "ground_truth.csv")):
            found.append(p)
    return found


def build_ground_truth_index(dataset_dirs):
    """content-hash(reference bytes + search bytes) -> {gt_x, gt_y, sample_id}.

    Reads the actual PNG files generate_dataset.py wrote and their actual
    ground_truth.csv rows -- the existing, unmodified metadata storage
    format -- and builds an in-memory lookup so an uploaded pair can be
    matched back to its real ground truth without the user typing it in."""
    index = {}
    for dataset_dir in dataset_dirs:
        try:
            df = pd.read_csv(os.path.join(dataset_dir, "ground_truth.csv"))
        except Exception:
            continue
        for _, row in df.iterrows():
            ref_path = os.path.join(dataset_dir, row["reference_file"])
            search_path = os.path.join(dataset_dir, row["search_file"])
            if not (os.path.isfile(ref_path) and os.path.isfile(search_path)):
                continue
            with open(ref_path, "rb") as f:
                ref_bytes = f.read()
            with open(search_path, "rb") as f:
                search_bytes = f.read()
            key = hashlib.sha256(ref_bytes + search_bytes).hexdigest()
            index[key] = {
                "gt_x": float(row["GT_X"]),
                "gt_y": float(row["GT_Y"]),
                "sample_id": str(row["sample_id"]),
            }
    return index


DATASET_DIRS = discover_dataset_dirs(REPO_ROOT)
GT_INDEX = build_ground_truth_index(DATASET_DIRS)
print(f"[drift-sense-demo] indexed {len(GT_INDEX)} known sample pairs "
      f"from {len(DATASET_DIRS)} dataset dir(s): {DATASET_DIRS}")


@app.post("/predict")
async def predict(
    reference: UploadFile = File(...),
    search: UploadFile = File(...),
):
    ref_bytes = await reference.read()
    search_bytes = await search.read()

    ref_gray = decode_upload_to_gray(ref_bytes)
    search_gray = decode_upload_to_gray(search_bytes)

    if ref_gray is None:
        return JSONResponse(status_code=400, content={
            "error": "Reference image could not be read. Unsupported or corrupt file."})
    if search_gray is None:
        return JSONResponse(status_code=400, content={
            "error": "Search image could not be read. Unsupported or corrupt file."})

    try:
        t0 = time.perf_counter()
        pred_x, pred_y = localize_cv_only(ref_gray, search_gray)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        # Never leak a raw stack trace to the frontend.
        return JSONResponse(status_code=500, content={
            "error": f"Prediction failed: {type(e).__name__}: {e}"})

    key = hashlib.sha256(ref_bytes + search_bytes).hexdigest()
    gt_entry = GT_INDEX.get(key)
    has_gt = gt_entry is not None

    pixel_error = None
    if has_gt:
        pixel_error = round(float(np.sqrt(
            (pred_x - gt_entry["gt_x"]) ** 2 + (pred_y - gt_entry["gt_y"]) ** 2
        )), 2)

    return {
        "mode": "classical CV + sub-pixel refinement (default, no-CNN, verified-best)",
        "pred_x": round(float(pred_x), 2),
        "pred_y": round(float(pred_y), 2),
        "has_gt": has_gt,
        "gt_x": gt_entry["gt_x"] if has_gt else None,
        "gt_y": gt_entry["gt_y"] if has_gt else None,
        "matched_sample_id": gt_entry["sample_id"] if has_gt else None,
        "pixel_error": pixel_error,
        "runtime_s": round(elapsed, 3),
    }


# Serve the frontend from the same app/port so there's one process to run.
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
