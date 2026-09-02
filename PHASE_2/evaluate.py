"""
Drift-Sense Phase 2 -- merged standalone evaluator
===================================================

Evaluates BOTH:
  * grayscale SEM sets (A/B/C) using the same classical-CV candidate
    generator as register.py
  * RGB optical sets (D/E) using the RGB matcher included below

The evaluator does NOT import register.py or register_rgb.py.

Usage examples
--------------
# Grayscale:
python eval\evaluate.py --datasets out_setA out_setB out_setC --split all --limit 150 --found-threshold 0.335 --out_dir eval\results

# Exactly 70 A + 70 B + 40 C from your local datasets:
python eval\evaluate.py --datasets out_setA out_setB out_setC --split all --limit-per-dataset 70 --found-threshold 0.335 --out_dir eval\results_210

# RGB:
python eval\evaluate.py --datasets out_setD --split all --limit 20 --found-threshold 0.335 --out_dir eval\results_rgb

Important:
  --limit limits the TOTAL number of rows after datasets are combined.
  --limit-per-dataset applies the same limit independently to each dataset.

Image type is detected automatically from the actual image:
  1 channel -> grayscale matcher
  3 channels -> RGB matcher

For grayscale, matching.py must be available in the project root because
that is the candidate-generation logic used by register.py.
"""

import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project import: same candidate generator used by register.py
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from matching import generate_candidates
except ImportError as exc:
    raise ImportError(
        "Could not import matching.generate_candidates. "
        "Place evaluate.py inside eval/ and keep matching.py in the project root."
    ) from exc


# ---------------------------------------------------------------------
# Common Phase 2 settings
# ---------------------------------------------------------------------

FOUND_THRESHOLD = 0.380 
CENTER_TIE_MARGIN = 0.03

LOCALIZATION_TIERS = [(1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40)]
SCALE_TIERS = [(1.0, 1.00), (2.0, 0.60), (5.0, 0.30)]
ROTATION_TIERS = [(0.25, 1.00), (0.50, 0.60), (1.00, 0.30)]


def localization_credit(err_px):
    for threshold, credit in LOCALIZATION_TIERS:
        if err_px <= threshold:
            return credit
    return 0.0


def tier_credit(error, tiers):
    for threshold, credit in tiers:
        if error <= threshold:
            return credit
    return 0.0


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_dataset_path(root, path):
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


# =====================================================================
# GRAYSCALE MATCHER
# =====================================================================

def localize_gray(reference, search):
    """
    Same selection logic as register.py:
      1. generate multi-scale/multi-rotation candidates
      2. find the best NCC score
      3. retain candidates within CENTER_TIE_MARGIN
      4. choose the candidate closest to the search-image centre
    """
    h, w = search.shape[:2]

    candidates = generate_candidates(reference, search, top_k=30)

    if not candidates:
        return w / 2.0, h / 2.0, 0.0, 10.0, 0.0

    search_center = (w / 2.0, h / 2.0)
    best_score = candidates[0][0]

    tied = [
        c for c in candidates
        if c[0] >= best_score - CENTER_TIE_MARGIN
    ]

    tied.sort(
        key=lambda c:
        (c[1] - search_center[0]) ** 2 +
        (c[2] - search_center[1]) ** 2
    )

    score, cx, cy, scale, angle = tied[0]

    return (
        float(cx),
        float(cy),
        float(angle),
        float(scale),
        float(score),
    )


# =====================================================================
# RGB MATCHER
# =====================================================================

RGB_SCALE_MIN = 8.0
RGB_SCALE_MAX = 12.0
RGB_SCALE_STEPS = 17

RGB_ROT_MIN = -5.0
RGB_ROT_MAX = 5.0
RGB_ROT_STEPS = 11

RGB_PEAKS_PER_MAP = 6
RGB_PEAK_SUPPRESS_RADIUS = 12
RGB_MIN_SCORE = 0.05


def rgb_feature(img):
    """
    Convert BGR RGB image to one scalar feature combining:
      - LAB luminance
      - weak chromatic information
      - gradient structure
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    l = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 127.0
    b = (lab[:, :, 2] - 128.0) / 127.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = cv2.magnitude(gx, gy)
    grad = grad / (grad.max() + 1e-6)

    feature = (
        l
        + 0.12 * np.sqrt(a * a + b * b)
        + 0.10 * (grad - 0.5)
    )

    return feature.astype(np.float32)


def rotate_feature(img, angle):
    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2.0, h / 2.0),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def local_maxima(corr, num_peaks, suppress_radius, min_score):
    peaks = []
    work = corr.copy()

    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)

        if max_val < min_score:
            break

        peaks.append((float(max_val), max_loc))

        # Suppress this peak so we can collect other local candidates.
        cv2.circle(
            work,
            max_loc,
            suppress_radius,
            -2.0,
            -1,
        )

    return peaks


def subpixel_offset(corr, loc):
    """
    Quadratic/parabolic interpolation around the integer correlation peak.
    Returns a subpixel correction in [-0.5, 0.5] for x and y.
    """
    x0, y0 = loc
    h, w = corr.shape[:2]

    dx = 0.0
    dy = 0.0

    if 0 < x0 < w - 1:
        fm1 = corr[y0, x0 - 1]
        f0 = corr[y0, x0]
        fp1 = corr[y0, x0 + 1]

        denom = fm1 - 2.0 * f0 + fp1

        if abs(denom) > 1e-9:
            dx = float(
                np.clip(
                    0.5 * (fm1 - fp1) / denom,
                    -0.5,
                    0.5,
                )
            )

    if 0 < y0 < h - 1:
        fm1 = corr[y0 - 1, x0]
        f0 = corr[y0, x0]
        fp1 = corr[y0 + 1, x0]

        denom = fm1 - 2.0 * f0 + fp1

        if abs(denom) > 1e-9:
            dy = float(
                np.clip(
                    0.5 * (fm1 - fp1) / denom,
                    -0.5,
                    0.5,
                )
            )

    return dx, dy


def generate_rgb_candidates(reference, search, top_k=30):
    ref_h, ref_w = reference.shape[:2]

    search_feature = rgb_feature(search)
    reference_feature = rgb_feature(reference)

    all_candidates = []

    for scale in np.linspace(
        RGB_SCALE_MIN,
        RGB_SCALE_MAX,
        RGB_SCALE_STEPS,
    ):
        new_w = max(8, int(round(ref_w / scale)))
        new_h = max(8, int(round(ref_h / scale)))

        if new_w >= search.shape[1] or new_h >= search.shape[0]:
            continue

        resized_ref = cv2.resize(
            reference_feature,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        for angle in np.linspace(
            RGB_ROT_MIN,
            RGB_ROT_MAX,
            RGB_ROT_STEPS,
        ):
            if abs(angle) > 1e-12:
                rotated = rotate_feature(resized_ref, angle)
            else:
                rotated = resized_ref

            result = cv2.matchTemplate(
                search_feature,
                rotated,
                cv2.TM_CCOEFF_NORMED,
            )

            peaks = local_maxima(
                result,
                RGB_PEAKS_PER_MAP,
                RGB_PEAK_SUPPRESS_RADIUS,
                RGB_MIN_SCORE,
            )

            for score, loc in peaks:
                dx, dy = subpixel_offset(result, loc)

                cx = (
                    loc[0]
                    + dx
                    + new_w / 2.0
                )

                cy = (
                    loc[1]
                    + dy
                    + new_h / 2.0
                )

                all_candidates.append(
                    (
                        float(score),
                        float(cx),
                        float(cy),
                        float(scale),
                        float(angle),
                    )
                )

    all_candidates.sort(key=lambda c: -c[0])

    # Remove duplicate spatial candidates before top_k.
    kept = []

    for candidate in all_candidates:
        score, cx, cy, scale, angle = candidate

        too_close = any(
            (cx - k[1]) ** 2 +
            (cy - k[2]) ** 2
            < RGB_PEAK_SUPPRESS_RADIUS ** 2
            for k in kept
        )

        if not too_close:
            kept.append(candidate)

        if len(kept) >= top_k:
            break

    return kept


def localize_rgb(reference, search):
    h, w = search.shape[:2]

    candidates = generate_rgb_candidates(
        reference,
        search,
        top_k=30,
    )

    if not candidates:
        return w / 2.0, h / 2.0, 0.0, 10.0, 0.0

    search_center = (w / 2.0, h / 2.0)

    best_score = candidates[0][0]

    tied = [
        c for c in candidates
        if c[0] >= best_score - CENTER_TIE_MARGIN
    ]

    tied.sort(
        key=lambda c:
        (c[1] - search_center[0]) ** 2 +
        (c[2] - search_center[1]) ** 2
    )

    score, cx, cy, scale, angle = tied[0]

    return (
        float(cx),
        float(cy),
        float(angle),
        float(scale),
        float(score),
    )


# =====================================================================
# DATASET LOADING
# =====================================================================

def load_combined(dataset_roots, limit_per_dataset=None):
    frames = []

    for root_idx, root in enumerate(dataset_roots):
        gt_path = os.path.join(root, "ground_truth.csv")

        if not os.path.isfile(gt_path):
            raise FileNotFoundError(
                f"ground_truth.csv not found: {gt_path}"
            )

        df = pd.read_csv(gt_path).copy()

        df["_root"] = root
        df["_root_idx"] = root_idx

        # Use sample_id/world_id when available. Otherwise pair_id.
        if "world_id" in df.columns:
            df["world_uid"] = (
                df["_root_idx"].astype(str)
                + "_"
                + df["world_id"].astype(str)
            )
        else:
            df["world_uid"] = (
                df["_root_idx"].astype(str)
                + "_world"
            )

        if "sample_id" in df.columns:
            df["sample_uid"] = (
                df["_root_idx"].astype(str)
                + "_"
                + df["sample_id"].astype(str)
            )
        elif "pair_id" in df.columns:
            df["sample_uid"] = (
                df["_root_idx"].astype(str)
                + "_"
                + df["pair_id"].astype(str)
            )
        else:
            df["sample_uid"] = (
                df["_root_idx"].astype(str)
                + "_"
                + df.index.astype(str)
            )

        if limit_per_dataset is not None:
            df = df.head(limit_per_dataset).copy()

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# =====================================================================
# EVALUATION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone merged grayscale + RGB Drift-Sense evaluator."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more dataset folders containing ground_truth.csv.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N combined rows.",
    )

    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Evaluate the first N rows independently from EACH dataset.",
    )

    parser.add_argument(
        "--split",
        choices=["all", "test"],
        default="all",
        help="Use all rows or a held-out test_split.csv.",
    )

    parser.add_argument(
        "--test_split_csv",
        default="test_split.csv",
        help="CSV containing sample_uid values for --split test.",
    )

    parser.add_argument(
        "--found-threshold",
        type=float,
        default=FOUND_THRESHOLD,
        help="NCC score threshold for found=1.",
    )

    parser.add_argument(
        "--out_dir",
        default="eval/results",
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional explicit detailed CSV path.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save submission-style predictions.csv.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image progress lines.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.out_csv is None:
        args.out_csv = os.path.join(
            args.out_dir,
            "manifest_merged.csv",
        )

    # ---------------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------------

    df = load_combined(
        args.datasets,
        limit_per_dataset=args.limit_per_dataset,
    )

    if df.empty:
        raise ValueError("No dataset rows found.")

    if args.split == "test":
        if not os.path.isfile(args.test_split_csv):
            raise FileNotFoundError(
                f"{args.test_split_csv} not found."
            )

        test_ids = set(
            pd.read_csv(args.test_split_csv)["sample_uid"]
            .astype(str)
        )

        df = df[
            df["sample_uid"].astype(str).isin(test_ids)
        ].reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    print()
    print("=" * 72)
    print("DRIFT-SENSE MERGED EVALUATOR")
    print("=" * 72)
    print(f"Datasets          : {', '.join(args.datasets)}")
    print(f"Pairs              : {len(df)}")
    print(f"Found threshold    : {args.found_threshold:.4f}")
    print(f"Python             : {platform.python_version()}")
    print(f"OpenCV             : {cv2.__version__}")
    print(f"Processor          : {platform.processor()}")
    print()
    print("Matcher selection  : automatic")
    print("  grayscale        -> matching.py / register.py CV logic")
    print("  RGB              -> embedded RGB CV matcher")
    print()

    rows = []
    times = []

    # ---------------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------------

    for i, row in df.iterrows():
        root = row["_root"]

        ref_col = find_column(
            df,
            ["reference_file", "reference_path", "reference"],
        )
        search_col = find_column(
            df,
            ["search_file", "search_path", "search"],
        )
        id_col = find_column(
            df,
            ["pair_id", "sample_id", "id"],
        )
        found_col = find_column(
            df,
            ["found", "true_found"],
        )

        if ref_col is None or search_col is None:
            raise ValueError(
                "ground_truth.csv needs reference_file/reference_path "
                "and search_file/search_path."
            )

        if id_col is None:
            raise ValueError(
                "ground_truth.csv needs pair_id or sample_id."
            )

        if found_col is None:
            raise ValueError(
                "ground_truth.csv needs found or true_found."
            )

        ref_path = resolve_dataset_path(
            root,
            row[ref_col],
        )
        search_path = resolve_dataset_path(
            root,
            row[search_col],
        )

        # Read unchanged first so we can automatically detect grayscale/RGB.
        reference_raw = cv2.imread(
            ref_path,
            cv2.IMREAD_UNCHANGED,
        )
        search_raw = cv2.imread(
            search_path,
            cv2.IMREAD_UNCHANGED,
        )

        if reference_raw is None:
            raise FileNotFoundError(
                f"Could not read reference: {ref_path}"
            )

        if search_raw is None:
            raise FileNotFoundError(
                f"Could not read search: {search_path}"
            )

        ref_channels = (
            1 if reference_raw.ndim == 2
            else reference_raw.shape[2]
        )
        search_channels = (
            1 if search_raw.ndim == 2
            else search_raw.shape[2]
        )

        if ref_channels == 1 and search_channels == 1:
            mode = "grayscale"
            reference = reference_raw
            search = search_raw
            matcher = localize_gray

        elif ref_channels == 3 and search_channels == 3:
            mode = "RGB"
            reference = reference_raw
            search = search_raw
            matcher = localize_rgb

        else:
            raise ValueError(
                f"Reference/search channel mismatch for "
                f"{row[id_col]}: ref={ref_channels}, "
                f"search={search_channels}"
            )

        t0 = time.perf_counter()

        pred_x, pred_y, pred_theta, pred_scale, score = matcher(
            reference,
            search,
        )

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        true_found = int(row[found_col])
        pred_found = int(score >= args.found_threshold)

        if pred_found:
            out_x = pred_x
            out_y = pred_y
            out_theta = pred_theta
            out_scale = pred_scale
        else:
            out_x = 0.0
            out_y = 0.0
            out_theta = 0.0
            out_scale = 0.0

        # GT values
        gt_x = (
            float(row["GT_X"])
            if "GT_X" in row and pd.notna(row["GT_X"])
            else np.nan
        )
        gt_y = (
            float(row["GT_Y"])
            if "GT_Y" in row and pd.notna(row["GT_Y"])
            else np.nan
        )

        if true_found and np.isfinite(gt_x) and np.isfinite(gt_y):
            pixel_error = float(
                np.hypot(
                    out_x - gt_x,
                    out_y - gt_y,
                )
            )

            loc_credit = localization_credit(pixel_error)
        else:
            pixel_error = np.nan
            loc_credit = np.nan

        # Pose metrics are optional because some local datasets may not
        # contain GT_theta_deg / GT_scale.
        if (
            true_found
            and "GT_scale" in row.index
            and "GT_theta_deg" in row.index
            and pd.notna(row["GT_scale"])
            and pd.notna(row["GT_theta_deg"])
        ):
            gt_scale = float(row["GT_scale"])
            gt_theta = float(row["GT_theta_deg"])

            scale_err_pct = (
                abs(out_scale - gt_scale)
                / max(1e-6, gt_scale)
                * 100.0
            )

            theta_err_deg = abs(
                out_theta - gt_theta
            )

            if loc_credit > 0:
                scale_credit = tier_credit(
                    scale_err_pct,
                    SCALE_TIERS,
                )
                rotation_credit = tier_credit(
                    theta_err_deg,
                    ROTATION_TIERS,
                )
            else:
                scale_credit = 0.0
                rotation_credit = 0.0
        else:
            gt_scale = np.nan
            gt_theta = np.nan
            scale_err_pct = np.nan
            theta_err_deg = np.nan
            scale_credit = np.nan
            rotation_credit = np.nan

        result_row = {
            "sample_uid": row["sample_uid"],
            "pair_id": row[id_col],
            "dataset": root,
            "mode": mode,
            "reference_file": str(row[ref_col]),
            "search_file": str(row[search_col]),

            "true_found": true_found,
            "pred_found": pred_found,
            "score": round(float(score), 6),

            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": round(float(out_x), 4),
            "pred_y": round(float(out_y), 4),
            "pixel_error": (
                round(float(pixel_error), 4)
                if np.isfinite(pixel_error)
                else np.nan
            ),
            "loc_credit": loc_credit,

            "gt_scale": gt_scale,
            "pred_scale": round(float(out_scale), 4),
            "scale_err_pct": (
                round(float(scale_err_pct), 4)
                if np.isfinite(scale_err_pct)
                else np.nan
            ),
            "scale_credit": scale_credit,

            "gt_theta_deg": gt_theta,
            "pred_theta_deg": round(float(out_theta), 4),
            "theta_err_deg": (
                round(float(theta_err_deg), 4)
                if np.isfinite(theta_err_deg)
                else np.nan
            ),
            "rotation_credit": rotation_credit,

            "runtime_s": round(float(elapsed), 4),
        }

        # Preserve useful generator metadata when present.
        for col in [
            "generation_mode",
            "difficulty_level_5_name",
            "architecture",
            "scale",
            "rotation",
            "noise_level",
            "seed",
        ]:
            if col in row.index:
                result_row[col] = row[col]

        rows.append(result_row)

        if not args.quiet:
            err_text = (
                f"{pixel_error:.3f}px"
                if np.isfinite(pixel_error)
                else "n/a"
            )

            print(
                f"[{i + 1:4d}/{len(df):4d}] "
                f"{mode:9s} "
                f"{'PRESENT' if true_found else 'ABSENT ':7s} "
                f"found={pred_found} "
                f"score={score:.3f} "
                f"err={err_text:>10s} "
                f"time={elapsed:.2f}s"
            )

    # ---------------------------------------------------------------
    # Save detailed manifest
    # ---------------------------------------------------------------

    result = pd.DataFrame(rows)
    result.to_csv(args.out_csv, index=False)

    print()
    print(f"Detailed manifest saved -> {args.out_csv}")

    if args.save_predictions:
        prediction_path = os.path.join(
            args.out_dir,
            "predictions.csv",
        )

        prediction_cols = [
            "pair_id",
            "pred_x",
            "pred_y",
            "pred_theta_deg",
            "pred_scale",
            "pred_found",
            "score",
        ]

        predictions = result[prediction_cols].copy()
        predictions.columns = [
            "pair_id",
            "x",
            "y",
            "theta",
            "scale",
            "found",
            "score",
        ]

        predictions.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"Submission-style predictions saved -> "
            f"{prediction_path}"
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    present = result[
        result["true_found"] == 1
    ].copy()

    absent = result[
        result["true_found"] == 0
    ].copy()

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(f"Total pairs       : {len(result)}")
    print(f"Present pairs     : {len(present)}")
    print(f"Absent pairs      : {len(absent)}")
    print(
        f"Predicted found   : "
        f"{int((result['pred_found'] == 1).sum())}"
    )
    print(
        f"Predicted reject  : "
        f"{int((result['pred_found'] == 0).sum())}"
    )

    # ---------------------------------------------------------------
    # Per-mode results
    # ---------------------------------------------------------------

    for mode, group in result.groupby("mode"):
        print()
        print(f"--- {mode.upper()} ---")
        print(f"Pairs: {len(group)}")

        group_present = group[
            group["true_found"] == 1
        ]

        if len(group_present):
            errs = (
                group_present["pixel_error"]
                .astype(float)
                .values
            )

            print(
                f"Mean error        : "
                f"{errs.mean():.4f} px"
            )
            print(
                f"Median error      : "
                f"{np.median(errs):.4f} px"
            )
            print(
                f"Worst error       : "
                f"{errs.max():.4f} px"
            )

            print()
            print(
                f"Subpixel <=0.50px : "
                f"{(errs <= 0.50).mean() * 100:.2f}%"
            )

            for threshold in [1.0, 2.0, 3.0, 5.0]:
                pct = (
                    errs <= threshold
                ).mean() * 100.0

                print(
                    f"Within {threshold:.0f}px       : "
                    f"{pct:.2f}%"
                )

            print(
                f"Mean loc credit   : "
                f"{group_present['loc_credit'].astype(float).mean():.4f}"
            )

        # Pose only if available.
        pose_rows = group_present[
            group_present["scale_credit"].notna()
        ]

        if len(pose_rows):
            print()
            print("--- POSE ---")
            print(
                f"Mean scale credit : "
                f"{pose_rows['scale_credit'].astype(float).mean():.4f}"
            )
            print(
                f"Mean rotation cr. : "
                f"{pose_rows['rotation_credit'].astype(float).mean():.4f}"
            )

    # ---------------------------------------------------------------
    # Rejection F1
    # ---------------------------------------------------------------

    if len(absent):
        y_true = result["true_found"].astype(int).values
        y_pred = result["pred_found"].astype(int).values

        tp = int(
            ((y_true == 1) & (y_pred == 1)).sum()
        )
        fp = int(
            ((y_true == 0) & (y_pred == 1)).sum()
        )
        tn = int(
            ((y_true == 0) & (y_pred == 0)).sum()
        )
        fn = int(
            ((y_true == 1) & (y_pred == 0)).sum()
        )

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2.0 * precision * recall
            / max(1e-12, precision + recall)
        )

        print()
        print("--- REJECTION ---")
        print(f"TP                : {tp}")
        print(f"FP                : {fp}")
        print(f"TN                : {tn}")
        print(f"FN                : {fn}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1                : {f1:.4f}")

        cm_path = os.path.join(
            args.out_dir,
            "rejection_confusion_matrix.csv",
        )

        pd.DataFrame(
            [
                [tn, fp],
                [fn, tp],
            ],
            index=["Actual Absent", "Actual Present"],
            columns=["Predicted Absent", "Predicted Present"],
        ).to_csv(cm_path)

        print(
            f"Confusion matrix saved -> {cm_path}"
        )
    else:
        print()
        print(
            "No absent rows in this run; "
            "rejection F1 was not computed."
        )

    # ---------------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------------

    rt = np.asarray(times, dtype=float)

    print()
    print("--- RUNTIME ---")
    print(
        f"Mean runtime      : "
        f"{rt.mean():.4f} s/pair"
    )
    print(
        f"Median runtime    : "
        f"{np.median(rt):.4f} s/pair"
    )
    print(
        f"Max runtime       : "
        f"{rt.max():.4f} s/pair"
    )
    print(
        f"5 sec median      : "
        f"{'PASS' if np.median(rt) <= 5.0 else 'OVER BUDGET'}"
    )

    # ---------------------------------------------------------------
    # Worst present case
    # ---------------------------------------------------------------

    if len(present):
        worst_idx = present["pixel_error"].astype(float).idxmax()
        worst = present.loc[worst_idx]

        search_path = resolve_dataset_path(
            worst["dataset"],
            worst["search_file"],
        )

        search = cv2.imread(
            search_path,
            cv2.IMREAD_COLOR,
        )

        if search is not None:
            gt = (
                int(round(float(worst["gt_x"]))),
                int(round(float(worst["gt_y"]))),
            )

            pred = (
                int(round(float(worst["pred_x"]))),
                int(round(float(worst["pred_y"]))),
            )

            # Green = GT, red = prediction.
            cv2.drawMarker(
                search,
                gt,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                24,
                2,
            )

            cv2.drawMarker(
                search,
                pred,
                (0, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                24,
                2,
            )

            failure_path = os.path.join(
                args.out_dir,
                f"failure_case_{worst['sample_uid']}.png",
            )

            cv2.imwrite(
                failure_path,
                search,
            )

            print()
            print(
                f"Worst case        : "
                f"{worst['sample_uid']}"
            )
            print(
                f"Worst error       : "
                f"{float(worst['pixel_error']):.4f}px"
            )
            print(
                f"Failure overlay    -> {failure_path}"
            )

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
