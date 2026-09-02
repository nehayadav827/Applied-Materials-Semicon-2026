"""
Drift-Sense Phase 2 -- standalone graded entry point
=====================================================

ONE ENTRY POINT
---------------
    python register.py --input pairs.csv --output predictions.csv

This file is fully standalone:
    - NO matching.py
    - NO matching_rgb.py
    - NO register_rgb.py
    - NO torch / GPU / network dependency

It automatically handles:
    * grayscale SEM pairs
    * RGB optical pairs

The search logic is intentionally kept the same as the supplied classical
CV implementation:

    multi-scale [8,12]
    multi-rotation [-5,+5]
    multiple local NCC peaks
    subpixel peak refinement
    nearest-to-centre tie-break
    local fine scale/rotation refinement

Output contract:
    pair_id,x,y,theta,scale,found,score

When found=0, x/y/theta/scale are all written as 0.

Relative image paths are resolved against --base_dir. If --base_dir is not
provided, it defaults to the directory containing --input.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Phase 2 constants
# ---------------------------------------------------------------------

SCALE_MIN = 8.0
SCALE_MAX = 12.0
SCALE_STEPS = 17

ROT_MIN = -5.0
ROT_MAX = 5.0
ROT_STEPS = 11

PEAKS_PER_MAP = 6
PEAK_SUPPRESS_RADIUS = 12
MIN_SCORE = 0.05

CENTER_TIE_MARGIN = 0.03

RUNTIME_SOFT_BUDGET_S = 5.0
RUNTIME_HARD_TIMEOUT_S = 20.0

FOUND_THRESHOLD = 0.380


# =====================================================================
# COMMON UTILITIES
# =====================================================================

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def resolve_path(base_dir, path):
    path = str(path)

    if os.path.isabs(path):
        return path

    return os.path.join(base_dir, path)


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"None of {candidates} found. "
        f"CSV columns: {list(df.columns)}"
    )


def load_image(path, label):
    """
    Load either grayscale or color without changing the source data.

    OpenCV reads ordinary RGB files as BGR, which is exactly what the RGB
    feature function below expects.

    Grayscale files are converted to a single channel so that the grayscale
    matcher remains identical to the original matching.py logic.
    """
    if not os.path.isfile(path):
        log(f"ERROR: {label} not found: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        log(f"ERROR: could not read {label}: {path}")
        return None

    # 2-D image = grayscale.
    if img.ndim == 2:
        return img

    # Normal 3-channel image.
    if img.ndim == 3 and img.shape[2] == 3:
        # Some PNG/TIFF files may technically be stored as 3 channels while
        # representing grayscale. Preserve the original grayscale path.
        if (
            np.all(img[:, :, 0] == img[:, :, 1])
            and np.all(img[:, :, 1] == img[:, :, 2])
        ):
            return img[:, :, 0]

        return img

    raise ValueError(
        f"Unsupported image shape for {label}: {img.shape}"
    )


# =====================================================================
# SHARED NCC HELPERS
# =====================================================================

def rotate_image(img, angle):
    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2.0, h / 2.0),
        float(angle),
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
    """
    Extract multiple spatially separated local correlation peaks.

    This is the same multiple-peak strategy used by the supplied
    matching.py implementation.
    """
    peaks = []
    work = corr.copy()

    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)

        if max_val < min_score:
            break

        peaks.append(
            (
                float(max_val),
                max_loc,
            )
        )

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
    Quadratic interpolation around the integer NCC maximum.

    Returns dx/dy in [-0.5, +0.5].
    """
    x0, y0 = loc
    h, w = corr.shape[:2]

    dx = 0.0
    dy = 0.0

    if 0 < x0 < w - 1:
        fm1 = float(corr[y0, x0 - 1])
        f0 = float(corr[y0, x0])
        fp1 = float(corr[y0, x0 + 1])

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
        fm1 = float(corr[y0 - 1, x0])
        f0 = float(corr[y0, x0])
        fp1 = float(corr[y0 + 1, x0])

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


# =====================================================================
# GRAYSCALE MATCHER
# =====================================================================

def generate_gray_candidates(reference, search, top_k=50):
    """
    Embedded version of matching.py::generate_candidates.

    IMPORTANT:
    This is intentionally kept at the same scale/rotation/peak settings
    as the supplied matcher, so register.py no longer depends on
    matching.py.
    """
    ref_h, ref_w = reference.shape[:2]

    all_candidates = []

    for scale in np.linspace(
        SCALE_MIN,
        SCALE_MAX,
        SCALE_STEPS,
    ):
        new_w = max(
            8,
            int(round(ref_w / scale)),
        )

        new_h = max(
            8,
            int(round(ref_h / scale)),
        )

        if (
            new_w >= search.shape[1]
            or new_h >= search.shape[0]
        ):
            continue

        resized_ref = cv2.resize(
            reference,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        for angle in np.linspace(
            ROT_MIN,
            ROT_MAX,
            ROT_STEPS,
        ):
            rotated = (
                rotate_image(
                    resized_ref,
                    float(angle),
                )
                if abs(float(angle)) > 1e-12
                else resized_ref
            )

            result = cv2.matchTemplate(
                search,
                rotated,
                cv2.TM_CCOEFF_NORMED,
            )

            for score, loc in local_maxima(
                result,
                PEAKS_PER_MAP,
                PEAK_SUPPRESS_RADIUS,
                MIN_SCORE,
            ):
                dx, dy = subpixel_offset(
                    result,
                    loc,
                )

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

    all_candidates.sort(
        key=lambda c: -c[0]
    )

    kept = []

    for cand in all_candidates:
        score, cx, cy, scale, angle = cand

        if not any(
            (cx - k[1]) ** 2
            + (cy - k[2]) ** 2
            < PEAK_SUPPRESS_RADIUS ** 2
            for k in kept
        ):
            kept.append(cand)

        if len(kept) >= top_k:
            break

    return kept


# =====================================================================
# RGB FEATURE + MATCHER
# =====================================================================

def robust01(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    lo, hi = np.percentile(
        values,
        (1, 99),
    )

    if hi - lo < 1e-6:
        return np.zeros_like(values)

    return np.clip(
        (values - lo) / (hi - lo),
        0.0,
        1.0,
    ).astype(np.float32)


def rgb_feature(img):
    """
    RGB/BGR -> scalar feature.

    Luminance remains dominant so the matcher stays close to the
    grayscale declared NCC method. A small chromatic-opponent component
    adds RGB information and a weak gradient component preserves texture.

    If a grayscale image reaches this function, it is normalized directly.
    """
    if img.ndim == 2:
        return (
            img.astype(np.float32)
            / 255.0
        )

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(
            f"Unsupported image shape: {img.shape}"
        )

    # OpenCV stores ordinary color images as BGR.
    lab = cv2.cvtColor(
        img[:, :, :3],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    L = lab[:, :, 0] / 255.0

    a = (
        lab[:, :, 1] - 128.0
    ) / 127.0

    b = (
        lab[:, :, 2] - 128.0
    ) / 127.0

    # Weak chromatic-opponent term.
    chroma = (
        0.7071 * a
        + 0.7071 * b
    )

    # Weak structural texture term.
    gx = cv2.Sobel(
        L,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        L,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    grad = cv2.magnitude(
        gx,
        gy,
    )

    grad = robust01(grad)

    feature = (
        L
        + 0.12 * chroma
        + 0.10 * (grad - 0.5)
    )

    return np.clip(
        feature,
        0.0,
        1.0,
    ).astype(np.float32)


def generate_rgb_candidates(reference, search, top_k=30):
    """
    Embedded version of the supplied standalone RGB matcher.
    """
    rf = rgb_feature(reference)
    sf = rgb_feature(search)

    ref_h, ref_w = rf.shape[:2]
    search_h, search_w = sf.shape[:2]

    all_candidates = []

    for scale in np.linspace(
        SCALE_MIN,
        SCALE_MAX,
        SCALE_STEPS,
    ):
        new_w = max(
            8,
            int(round(ref_w / scale)),
        )

        new_h = max(
            8,
            int(round(ref_h / scale)),
        )

        if (
            new_w >= search_w
            or new_h >= search_h
        ):
            continue

        resized_ref = cv2.resize(
            rf,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        for angle in np.linspace(
            ROT_MIN,
            ROT_MAX,
            ROT_STEPS,
        ):
            if abs(float(angle)) > 1e-12:
                rotated = rotate_image(
                    resized_ref,
                    float(angle),
                )
            else:
                rotated = resized_ref

            corr = cv2.matchTemplate(
                sf,
                rotated,
                cv2.TM_CCOEFF_NORMED,
            )

            peaks = local_maxima(
                corr,
                PEAKS_PER_MAP,
                PEAK_SUPPRESS_RADIUS,
                MIN_SCORE,
            )

            for score, loc in peaks:
                dx, dy = subpixel_offset(
                    corr,
                    loc,
                )

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

    all_candidates.sort(
        key=lambda c: -c[0]
    )

    kept = []

    for cand in all_candidates:
        score, cx, cy, scale, angle = cand

        too_close = any(
            (cx - k[1]) ** 2
            + (cy - k[2]) ** 2
            < PEAK_SUPPRESS_RADIUS ** 2
            for k in kept
        )

        if not too_close:
            kept.append(cand)

        if len(kept) >= top_k:
            break

    return kept


# =====================================================================
# POSE REFINEMENT
# =====================================================================

def refine_pose(reference, search, candidate):
    """
    Fine local scale/rotation refinement around the already selected tile.

    This is the same refinement logic from the supplied register.py.
    It does NOT change the search family: it remains local NCC.
    """
    base_score, cx0, cy0, s0, a0 = candidate

    color = (
        reference.ndim == 3
        or search.ndim == 3
    )

    if color:
        ref_f = rgb_feature(reference)
        sea_f = rgb_feature(search)
    else:
        ref_f = (
            reference.astype(np.float32)
            / 255.0
        )

        sea_f = (
            search.astype(np.float32)
            / 255.0
        )

    best = candidate

    # Fine scale refinement around coarse candidate.
    for scale in np.arange(
        max(SCALE_MIN, s0 - 0.20),
        min(SCALE_MAX, s0 + 0.20) + 1e-9,
        0.10,
    ):
        nw = max(
            8,
            int(round(
                ref_f.shape[1] / scale
            )),
        )

        nh = max(
            8,
            int(round(
                ref_f.shape[0] / scale
            )),
        )

        tmpl0 = cv2.resize(
            ref_f,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

        half_w = nw / 2.0 + 12.0
        half_h = nh / 2.0 + 12.0

        x0 = max(
            0,
            int(round(cx0 - half_w)),
        )

        y0 = max(
            0,
            int(round(cy0 - half_h)),
        )

        x1 = min(
            sea_f.shape[1],
            int(round(cx0 + half_w)),
        )

        y1 = min(
            sea_f.shape[0],
            int(round(cy0 + half_h)),
        )

        roi = sea_f[
            y0:y1,
            x0:x1,
        ]

        if (
            roi.shape[0] <= nh
            or roi.shape[1] <= nw
        ):
            continue

        # Fine rotation refinement.
        for angle in np.arange(
            max(ROT_MIN, a0 - 0.50),
            min(ROT_MAX, a0 + 0.50) + 1e-9,
            0.25,
        ):
            if abs(float(angle)) > 1e-12:
                tmpl = rotate_image(
                    tmpl0,
                    float(angle),
                )
            else:
                tmpl = tmpl0

            corr = cv2.matchTemplate(
                roi,
                tmpl,
                cv2.TM_CCOEFF_NORMED,
            )

            _, score, _, loc = cv2.minMaxLoc(
                corr
            )

            dx, dy = subpixel_offset(
                corr,
                loc,
            )

            cx = (
                x0
                + loc[0]
                + dx
                + nw / 2.0
            )

            cy = (
                y0
                + loc[1]
                + dy
                + nh / 2.0
            )

            cand = (
                float(score),
                float(cx),
                float(cy),
                float(scale),
                float(angle),
            )

            if cand[0] > best[0]:
                best = cand

    return best


# =====================================================================
# REGISTRATION
# =====================================================================

def register_one(reference, search):
    """
    Automatically chooses grayscale or RGB matcher from image channels.
    """
    h, w = search.shape[:2]

    is_color = (
        reference.ndim == 3
        or search.ndim == 3
    )

    if is_color:
        candidates = generate_rgb_candidates(
            reference,
            search,
            top_k=30,
        )
    else:
        candidates = generate_gray_candidates(
            reference,
            search,
            top_k=30,
        )

    if not candidates:
        return (
            w / 2.0,
            h / 2.0,
            0.0,
            10.0,
            -1.0,
        )

    search_center = (
        w / 2.0,
        h / 2.0,
    )

    best_score = candidates[0][0]

    # Phase-2 nearest-to-centre tie-break.
    tied = [
        c
        for c in candidates
        if c[0]
        >= best_score - CENTER_TIE_MARGIN
    ]

    tied.sort(
        key=lambda c:
        (c[1] - search_center[0]) ** 2
        + (c[2] - search_center[1]) ** 2
    )

    # Fine pose refinement around the selected tile.
    chosen = refine_pose(
        reference,
        search,
        tied[0],
    )

    score, cx, cy, scale, angle = chosen

    return (
        float(cx),
        float(cy),
        float(angle),
        float(scale),
        float(score),
    )


# =====================================================================
# CONFIDENCE
# =====================================================================

def decision_confidence(
    raw_score,
    found,
    threshold,
):
    """
    Preserve the supplied register.py confidence convention.

    Larger score means greater confidence in the final found/rejected
    decision. The binary decision itself is always made from raw NCC
    against the threshold.
    """
    if found:
        return float(
            raw_score - threshold
        )

    return float(
        threshold - raw_score
    )


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Drift-Sense Phase 2 "
            "grayscale + RGB registration."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="pairs.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="predictions.csv",
    )

    parser.add_argument(
        "--base_dir",
        default=None,
        help=(
            "Base directory for relative image paths. "
            "Default: directory containing --input."
        ),
    )

    parser.add_argument(
        "--found-threshold",
        type=float,
        default=FOUND_THRESHOLD,
        help=(
            f"NCC threshold for found=1. "
            f"Default: {FOUND_THRESHOLD}"
        ),
    )

    args = parser.parse_args()

    base_dir = (
        args.base_dir
        or os.path.dirname(
            os.path.abspath(args.input)
        )
        or "."
    )

    pairs = pd.read_csv(
        args.input
    )

    id_col = find_column(
        pairs,
        [
            "pair_id",
            "sample_id",
        ],
    )

    ref_col = find_column(
        pairs,
        [
            "reference_path",
            "reference_file",
            "reference",
        ],
    )

    search_col = find_column(
        pairs,
        [
            "search_path",
            "search_file",
            "search",
        ],
    )

    log(
        f"Loaded {len(pairs)} pairs "
        f"from {args.input}"
    )

    log(
        f"found threshold = "
        f"{args.found_threshold:.4f}"
    )

    rows = []
    runtimes = []

    for index, row in pairs.iterrows():
        pair_id = row[id_col]

        ref_path = resolve_path(
            base_dir,
            row[ref_col],
        )

        search_path = resolve_path(
            base_dir,
            row[search_col],
        )

        t0 = time.perf_counter()

        reference = load_image(
            ref_path,
            "reference",
        )

        search = load_image(
            search_path,
            "search",
        )

        if (
            reference is None
            or search is None
        ):
            rows.append(
                {
                    "pair_id": pair_id,
                    "x": 0,
                    "y": 0,
                    "theta": 0,
                    "scale": 0,
                    "found": 0,
                    "score": 0.0,
                }
            )

            log(
                f"{pair_id}: unreadable "
                f"image -> found=0"
            )

            continue

        # Make sure both images are either grayscale or color.
        ref_color = (
            reference.ndim == 3
        )

        search_color = (
            search.ndim == 3
        )

        if ref_color != search_color:
            log(
                f"{pair_id}: reference/search "
                f"channel mismatch -> found=0"
            )

            rows.append(
                {
                    "pair_id": pair_id,
                    "x": 0,
                    "y": 0,
                    "theta": 0,
                    "scale": 0,
                    "found": 0,
                    "score": 0.0,
                }
            )

            continue

        try:
            (
                cx,
                cy,
                theta,
                scale,
                raw_score,
            ) = register_one(
                reference,
                search,
            )

        except Exception as exc:
            log(
                f"{pair_id}: registration error: "
                f"{exc} -> found=0"
            )

            rows.append(
                {
                    "pair_id": pair_id,
                    "x": 0,
                    "y": 0,
                    "theta": 0,
                    "scale": 0,
                    "found": 0,
                    "score": 0.0,
                }
            )

            continue

        elapsed = (
            time.perf_counter()
            - t0
        )

        runtimes.append(
            elapsed
        )

        found = int(
            raw_score
            >= args.found_threshold
        )

        confidence = decision_confidence(
            raw_score,
            found,
            args.found_threshold,
        )

        # Phase-2 contract:
        # pose columns must be zero when found=0.
        if found == 0:
            result = {
                "pair_id": pair_id,
                "x": 0,
                "y": 0,
                "theta": 0,
                "scale": 0,
                "found": 0,
                "score": round(
                    confidence,
                    6,
                ),
            }

        else:
            result = {
                "pair_id": pair_id,
                "x": round(
                    float(cx),
                    3,
                ),
                "y": round(
                    float(cy),
                    3,
                ),
                "theta": round(
                    float(theta),
                    4,
                ),
                "scale": round(
                    float(scale),
                    5,
                ),
                "found": 1,
                "score": round(
                    confidence,
                    6,
                ),
            }

        rows.append(result)

        budget_flag = ""

        if elapsed > RUNTIME_HARD_TIMEOUT_S:
            budget_flag = (
                " [OVER 20s HARD TIMEOUT]"
            )
        elif elapsed > RUNTIME_SOFT_BUDGET_S:
            budget_flag = (
                " [over 5s soft budget]"
            )

        mode = (
            "RGB"
            if ref_color
            else "GRAY"
        )

        log(
            f"{pair_id}: "
            f"mode={mode} "
            f"found={found} "
            f"raw={raw_score:.4f} "
            f"score={confidence:.4f} "
            f"time={elapsed:.2f}s"
            f"{budget_flag}"
        )

    # ---------------------------------------------------------------
    # Exact required output column order.
    # ---------------------------------------------------------------

    out = pd.DataFrame(
        rows,
        columns=[
            "pair_id",
            "x",
            "y",
            "theta",
            "scale",
            "found",
            "score",
        ],
    )

    out.to_csv(
        args.output,
        index=False,
    )

    log(
        f"\nWrote {len(out)} predictions "
        f"-> {args.output}"
    )

    if runtimes:
        median_runtime = float(
            np.median(runtimes)
        )

        mean_runtime = float(
            np.mean(runtimes)
        )

        max_runtime = float(
            np.max(runtimes)
        )

        log(
            "Runtime: "
            f"median={median_runtime:.2f}s "
            f"mean={mean_runtime:.2f}s "
            f"max={max_runtime:.2f}s"
        )

        if median_runtime > RUNTIME_SOFT_BUDGET_S:
            log(
                "WARNING: median runtime "
                "is above the 5s soft budget."
            )

        if max_runtime > RUNTIME_HARD_TIMEOUT_S:
            log(
                "WARNING: at least one pair "
                "exceeded the 20s hard timeout."
            )


if __name__ == "__main__":
    main()
