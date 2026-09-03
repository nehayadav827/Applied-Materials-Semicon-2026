"""
Drift-Sense Phase 2 -- results dashboard + live demo backend.

Two things live here:

1. GET /summary -- reads eval/results/manifest_merged.csv (written by
   PHASE_2/evaluate.py) and computes the addendum's exact Phase 2 score
   breakdown (Slide 5): Localization 40, Pose 20, Rejection 15,
   Calibration 10, plus Efficiency/Bonus context. The manifest already
   contains per-pair loc_credit/scale_credit/rotation_credit/true_found/
   pred_found -- this endpoint only AGGREGATES those with the addendum's
   published formulas, it never re-derives correctness itself. Also
   returns per-set (A/B/C/D), per-generation_mode, and per-difficulty-tier
   breakdowns, plus the worst-case failure pair (and its saved image, if
   evaluate.py wrote one to eval/results/).

2. POST /predict -- a live single-pair demo. Imports register_one(),
   decision_confidence(), and FOUND_THRESHOLD directly from the ACTUAL
   graded register.py in this folder -- no prediction logic is
   duplicated here, so a demo result is exactly what a real predictions.csv
   row for that pair would contain. Ground truth is only ever shown if the
   uploaded reference+search bytes hash-match a known sample from a
   ground_truth.csv this app can find on disk (same "never fabricate GT"
   principle as the Phase 1 demo) -- never fabricated or guessed.

Run from PHASE_2/frontend/:
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000
"""

import glob
import hashlib
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# register.py lives in PHASE_2/, one level up from PHASE_2/frontend/.
PHASE2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PHASE2_ROOT)
sys.path.insert(0, PHASE2_ROOT)

# The ACTUAL graded pipeline -- nothing about prediction is reimplemented
# in this file. If register.py changes, this demo changes with it.
from register import register_one, decision_confidence, FOUND_THRESHOLD  # noqa: E402

DEFAULT_MANIFEST = os.path.join(PHASE2_ROOT, "eval", "results", "manifest_merged.csv")
RESULTS_DIR = os.path.join(PHASE2_ROOT, "eval", "results")

app = FastAPI(title="Drift-Sense Phase 2 Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Ground-truth auto-discovery (same principle as the Phase 1 demo: only
# ever show GT that was actually generated on disk, hash-matched to the
# uploaded bytes -- never fabricated for an arbitrary uploaded image).
# =====================================================================

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def discover_dataset_dirs():
    """Any directory (searched a couple of levels under PHASE_2/ and the
    repo root) containing a ground_truth.csv is a dataset someone's
    generate_dataset.py produced."""
    found = []
    search_roots = [PHASE2_ROOT, REPO_ROOT]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for gt_path in glob.glob(os.path.join(root, "**", "ground_truth.csv"), recursive=True):
            d = os.path.dirname(gt_path)
            if d not in found:
                found.append(d)
    extra = os.environ.get("DRIFT_SENSE_DATASET_DIRS", "")
    for p in extra.split(os.pathsep):
        p = p.strip()
        if p and os.path.isfile(os.path.join(p, "ground_truth.csv")):
            found.append(p)
    return found


def build_ground_truth_index(dataset_dirs):
    """content-hash(reference bytes + search bytes) -> GT fields, read
    straight from each dataset's own ground_truth.csv + PNG files."""
    index = {}
    for dataset_dir in dataset_dirs:
        try:
            df = pd.read_csv(os.path.join(dataset_dir, "ground_truth.csv"))
        except Exception:
            continue
        id_col = _find_col(df, ["pair_id", "sample_id"])
        ref_col = _find_col(df, ["reference_path", "reference_file"])
        search_col = _find_col(df, ["search_path", "search_file"])
        found_col = _find_col(df, ["found", "true_instance_present"])
        if not (id_col and ref_col and search_col):
            continue
        for _, row in df.iterrows():
            ref_path = os.path.join(dataset_dir, str(row[ref_col]))
            search_path = os.path.join(dataset_dir, str(row[search_col]))
            if not (os.path.isfile(ref_path) and os.path.isfile(search_path)):
                continue
            try:
                with open(ref_path, "rb") as f:
                    ref_bytes = f.read()
                with open(search_path, "rb") as f:
                    search_bytes = f.read()
            except OSError:
                continue
            key = hashlib.sha256(ref_bytes + search_bytes).hexdigest()
            is_present = bool(row[found_col]) if found_col else True
            entry = {
                "sample_id": str(row[id_col]),
                "source": os.path.basename(dataset_dir),
                "is_present": is_present,
            }
            if is_present and "GT_X" in df.columns and "GT_Y" in df.columns:
                entry["gt_x"] = float(row["GT_X"])
                entry["gt_y"] = float(row["GT_Y"])
                if "GT_theta_deg" in df.columns:
                    entry["gt_theta"] = float(row["GT_theta_deg"])
                if "GT_scale" in df.columns:
                    entry["gt_scale"] = float(row["GT_scale"])
            index[key] = entry
    return index


DATASET_DIRS = discover_dataset_dirs()
GT_INDEX = build_ground_truth_index(DATASET_DIRS)
print(f"[phase2-dashboard] indexed {len(GT_INDEX)} known sample pairs "
      f"from {len(DATASET_DIRS)} dataset dir(s) under {PHASE2_ROOT} / {REPO_ROOT}")


@app.post("/reindex")
def reindex():
    """Rebuilds the ground-truth index without restarting the server --
    call this after generating a new dataset folder so /predict can find
    it immediately (the index is otherwise only built once at startup)."""
    global DATASET_DIRS, GT_INDEX
    DATASET_DIRS = discover_dataset_dirs()
    GT_INDEX = build_ground_truth_index(DATASET_DIRS)
    print(f"[phase2-dashboard] reindexed: {len(GT_INDEX)} known sample pairs "
          f"from {len(DATASET_DIRS)} dataset dir(s)")
    return {"n_samples_indexed": len(GT_INDEX), "n_dataset_dirs": len(DATASET_DIRS),
            "dataset_dirs": [os.path.basename(d) for d in DATASET_DIRS]}


@app.get("/status")
def status():
    return {"n_samples_indexed": len(GT_INDEX), "n_dataset_dirs": len(DATASET_DIRS),
            "dataset_dirs": [os.path.basename(d) for d in DATASET_DIRS]}


# =====================================================================
# /predict -- live single-pair demo, delegating entirely to register.py
# =====================================================================

def decode_upload(raw_bytes):
    """Mirrors register.py's load_image(), from in-memory bytes instead
    of a file path -- same grayscale/RGB auto-detection, same
    equal-channel-collapse-to-grayscale behaviour."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:
        if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
            return img[:, :, 0]
    return img


@app.post("/predict")
async def predict(reference: UploadFile = File(...), search: UploadFile = File(...)):
    ref_bytes = await reference.read()
    search_bytes = await search.read()

    ref_img = decode_upload(ref_bytes)
    search_img = decode_upload(search_bytes)

    if ref_img is None:
        return JSONResponse(status_code=400, content={
            "error": "Reference image could not be read. Unsupported or corrupt file."})
    if search_img is None:
        return JSONResponse(status_code=400, content={
            "error": "Search image could not be read. Unsupported or corrupt file."})

    ref_color = ref_img.ndim == 3
    search_color = search_img.ndim == 3
    if ref_color != search_color:
        # Mirrors register.py's own channel-mismatch handling: found=0,
        # not a hard error -- a real submission run would score this
        # pair zero, not crash.
        return {
            "mode": "channel-mismatch",
            "x": 0, "y": 0, "theta": 0, "scale": 0,
            "found": 0, "score": 0.0,
            "found_threshold": FOUND_THRESHOLD,
            "note": "Reference and search images have different channel counts "
                    "(one grayscale, one RGB) -- register.py would reject this pair.",
            "has_gt": False, "runtime_s": 0.0,
        }

    try:
        t0 = time.perf_counter()
        cx, cy, theta, scale, raw_score = register_one(ref_img, search_img)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": f"Registration failed: {type(e).__name__}: {e}"})

    found = int(raw_score >= FOUND_THRESHOLD)
    confidence = decision_confidence(raw_score, found, FOUND_THRESHOLD)

    key = hashlib.sha256(ref_bytes + search_bytes).hexdigest()
    gt_entry = GT_INDEX.get(key)
    has_gt = gt_entry is not None

    pixel_error = None
    if has_gt and found == 1 and gt_entry.get("is_present") and "gt_x" in gt_entry:
        pixel_error = round(float(np.hypot(cx - gt_entry["gt_x"], cy - gt_entry["gt_y"])), 3)

    result = {
        "mode": "RGB" if ref_color else "grayscale",
        "found": found,
        "found_threshold": round(float(FOUND_THRESHOLD), 4),
        "raw_ncc_score": round(float(raw_score), 4),
        "score": round(float(confidence), 6),
        "runtime_s": round(elapsed, 3),
        "has_gt": has_gt,
    }

    # Output-contract fields: zeroed when found=0, exactly as
    # predictions.csv would contain per the addendum ("when found=0, x/y/
    # theta/scale are all written as 0").
    if found == 1:
        result.update(x=round(float(cx), 3), y=round(float(cy), 3),
                       theta=round(float(theta), 4), scale=round(float(scale), 5))
    else:
        result.update(x=0, y=0, theta=0, scale=0)
        # Diagnostic-only, NOT part of the official output contract --
        # clearly separated so the UI can label it as such.
        result["diagnostic_raw_xy"] = {"x": round(float(cx), 3), "y": round(float(cy), 3)}

    if has_gt:
        result["matched_sample_id"] = gt_entry["sample_id"]
        result["matched_source"] = gt_entry["source"]
        result["gt_is_present"] = gt_entry.get("is_present", True)
        if "gt_x" in gt_entry:
            result["gt_x"] = gt_entry["gt_x"]
            result["gt_y"] = gt_entry["gt_y"]
            result["gt_theta"] = gt_entry.get("gt_theta")
            result["gt_scale"] = gt_entry.get("gt_scale")
        result["pixel_error"] = pixel_error

    return result


# =====================================================================
# /summary -- aggregate Phase 2 score breakdown from manifest_merged.csv
# =====================================================================

def _extract_set(dataset_value):
    m = re.search(r"out_set([A-Za-z])", str(dataset_value), re.IGNORECASE)
    return m.group(1).upper() if m else None


@app.get("/summary")
def summary(manifest: str = None):
    manifest_path = manifest or DEFAULT_MANIFEST
    if not os.path.isfile(manifest_path):
        return JSONResponse(status_code=404, content={
            "error": f"No manifest found at {manifest_path}. Run evaluate.py first, e.g.:\n"
                     f"  python evaluate.py --datasets ./out_setA_new ./out_setB_new "
                     f"./out_setC_new ./out_setD_new --out_dir eval/results"})

    df = pd.read_csv(manifest_path)
    df["set"] = df["dataset"].apply(_extract_set)
    if df["set"].isna().all():
        # Fallback bucketing if the dataset column doesn't follow the
        # out_set<X> convention this run used -- group by mode instead so
        # the dashboard still renders something meaningful.
        df["set"] = df["mode"].map({"grayscale": "A/B/C (grayscale)", "RGB": "D (RGB)"})

    present = df[df["true_found"] == 1].copy()

    # ---- Localization (40 pts): 0.45*A + 0.55*B, scaled to 40 ----
    set_scores = {}
    for s in ("A", "B"):
        sub = present[present["set"] == s]
        set_scores[s] = float(sub["loc_credit"].mean()) if len(sub) else None
    if set_scores.get("A") is not None and set_scores.get("B") is not None:
        loc_total = (0.45 * set_scores["A"] + 0.55 * set_scores["B"]) * 40
    else:
        loc_total = None

    # ---- Pose (20 pts): only where localization credit > 0 ----
    loc_correct = present[present["loc_credit"] > 0]
    if len(loc_correct):
        scale_total = float(loc_correct["scale_credit"].mean()) * 10
        rot_total = float(loc_correct["rotation_credit"].mean()) * 10
        pose_total = scale_total + rot_total
    else:
        scale_total = rot_total = pose_total = None

    # ---- Rejection (15 pts): F1 on found flag, A+B+C grayscale only ----
    grayscale = df[df["set"].isin(["A", "B", "C"])]
    rejection_f1 = None
    confusion = None
    if len(grayscale):
        tp = int(((grayscale["true_found"] == 1) & (grayscale["pred_found"] == 1)).sum())
        fp = int(((grayscale["true_found"] == 0) & (grayscale["pred_found"] == 1)).sum())
        fn = int(((grayscale["true_found"] == 1) & (grayscale["pred_found"] == 0)).sum())
        tn = int(((grayscale["true_found"] == 0) & (grayscale["pred_found"] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rejection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        confusion = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    rejection_total = (rejection_f1 * 15) if rejection_f1 is not None else None

    # ---- Confidence calibration (10 pts): AUC of score vs correctness ----
    correctness = ((df["true_found"] == df["pred_found"]) &
                   ((df["true_found"] == 0) | (df["pixel_error"] <= 5))).astype(int)
    calib_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if correctness.nunique() > 1:
            calib_auc = float(roc_auc_score(correctness, df["score"]))
    except Exception:
        calib_auc = None
    calib_total = (calib_auc * 10) if calib_auc is not None else None

    core_total = sum(v for v in [loc_total, pose_total, rejection_total, calib_total] if v is not None)

    # ---- Efficiency (5 pts): NOT self-scorable (relative quartile vs.
    # other teams per the addendum) -- report own runtime only ----
    median_runtime = float(df["runtime_s"].median()) if "runtime_s" in df.columns else None

    # ---- Bonus (+10): Set D credit >=0.40 with A-C >=0.50, +4 if F1>=0.90 ----
    set_d_present = present[present["set"] == "D"]
    set_d_credit = float(set_d_present["loc_credit"].mean()) if len(set_d_present) else None
    abc_present = present[present["set"].isin(["A", "B", "C"])]
    abc_credit = float(abc_present["loc_credit"].mean()) if len(abc_present) else None

    # ---- Per-set table ----
    per_set = []
    for s in sorted(df["set"].dropna().unique()):
        sub = df[df["set"] == s]
        row = {"set": s, "n": int(len(sub)), "avg_score": round(float(sub["score"].mean()), 3)}
        if s == "C":
            correctly_rejected = int((sub["pred_found"] == 0).sum())
            row["correctly_rejected"] = correctly_rejected
            row["rejection_pct"] = round(correctly_rejected / len(sub) * 100, 1) if len(sub) else None
        else:
            sub_present = present[present["set"] == s]
            row["pred_found_1"] = int((sub["pred_found"] == 1).sum())
            row["mean_err_px"] = round(float(sub_present["pixel_error"].mean()), 2) if len(sub_present) else None
            row["within_5px_pct"] = round(float((sub_present["pixel_error"] <= 5).mean() * 100), 1) if len(sub_present) else None
        per_set.append(row)

    # ---- Per-generation_mode / per-difficulty-tier (present only) ----
    def _bucket(colname):
        if colname not in present.columns:
            return []
        out = []
        for name, grp in present.groupby(colname):
            out.append({
                "name": str(name), "n": int(len(grp)),
                "mean_err_px": round(float(grp["pixel_error"].mean()), 2),
                "within_5px_pct": round(float((grp["pixel_error"] <= 5).mean() * 100), 1),
            })
        return sorted(out, key=lambda r: r["name"])

    by_generation_mode = _bucket("generation_mode")
    by_difficulty_tier = _bucket("difficulty_level_5_name")

    # ---- Worst case + failure image, if evaluate.py saved one ----
    worst_case = None
    if len(present):
        worst_row = present.loc[present["pixel_error"].idxmax()]
        image_url = None
        candidates = glob.glob(os.path.join(RESULTS_DIR, f"failure_case_*{worst_row['pair_id']}*.png"))
        if candidates:
            image_url = "/results/" + os.path.basename(candidates[0])
        worst_case = {
            "pair_id": str(worst_row["pair_id"]),
            "set": worst_row["set"],
            "error_px": round(float(worst_row["pixel_error"]), 2),
            "image_url": image_url,
        }

    return {
        "manifest_path": manifest_path,
        "n_rows": int(len(df)),
        "localization": {"a_credit": set_scores.get("A"), "b_credit": set_scores.get("B"),
                          "score": loc_total, "max": 40},
        "pose": {"scale_score": scale_total, "rotation_score": rot_total,
                 "score": pose_total, "max": 20,
                 "n_eligible": int(len(loc_correct)), "n_present": int(len(present))},
        "rejection": {"f1": rejection_f1, "score": rejection_total, "max": 15,
                      "confusion": confusion, "n_grayscale": int(len(grayscale))},
        "calibration": {"auc": calib_auc, "score": calib_total, "max": 10},
        "core_total": {"score": core_total, "max": 90},
        "efficiency": {"median_runtime_s": median_runtime, "soft_budget_s": 5.0,
                       "note": "Efficiency (5 pts) is scored by relative quartile rank against "
                               "other teams' median runtime -- not self-computable. Shown here "
                               "for your own budget sanity check only."},
        "bonus": {"set_d_credit": set_d_credit, "set_d_threshold": 0.40,
                  "abc_credit": abc_credit, "abc_threshold": 0.50,
                  "rejection_f1_threshold": 0.90,
                  "set_d_bonus_eligible": (set_d_credit is not None and abc_credit is not None
                                            and set_d_credit >= 0.40 and abc_credit >= 0.50),
                  "rejection_bonus_eligible": (rejection_f1 is not None and rejection_f1 >= 0.90)},
        "per_set": per_set,
        "by_generation_mode": by_generation_mode,
        "by_difficulty_tier": by_difficulty_tier,
        "worst_case": worst_case,
    }


# Serve saved failure-case images (and anything else evaluate.py writes)
# from eval/results/, and the dashboard frontend at "/".
if os.path.isdir(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")