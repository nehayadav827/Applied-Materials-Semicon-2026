"""
Shared candidate-generation logic for Drift-Sense Phase 2. Extracts
MULTIPLE local peaks per scale/rotation combination, not just the single
global max -- this is the fix for low recall on periodic patterns.

Search ranges widened for Phase 2's unknown-pose addendum:
  - scale [8, 12]x  (was fixed 10x in Phase 1)
  - rotation +/-5deg (was +/-2deg tuned for Phase 1's 1-3deg noise-only case)

This is the version used by register.py (the graded entry point). An
alternative coarse-to-fine search (matching_v2.py, kept in training/ as an
ablation) was tried and measured WORSE on held-out eval (14.4% within 5px
vs this file's 17.8%) despite being faster -- so this flat-grid version is
the one actually shipped, even though it runs closer to the runtime budget.
See failure_analysis.md for the full comparison.
"""

import cv2
import numpy as np

SCALE_MIN, SCALE_MAX, SCALE_STEPS = 8.0, 12.0, 17   # 0.25 step across the full disclosed range
ROT_MIN, ROT_MAX, ROT_STEPS = -5.0, 5.0, 11          # 1.0 step across the full disclosed range
PEAKS_PER_MAP = 6
PEAK_SUPPRESS_RADIUS = 12
MIN_SCORE = 0.05
# NOTE: do NOT pre-reject weak-but-correct peaks before selection sees them;
# let nearest-to-centre tie-breaking (in register.py/evaluate.py) filter
# instead of a fixed NCC floor. The dominant recall loss on this dataset is
# NOT a tunable-parameter problem: single_field-mode samples (one
# continuous periodic field, no landmark) are close to 0% recall by design
# -- genuinely ambiguous, matching the "several tiles match" scenario the
# problem statement itself describes. legacy/realistic modes (grid +
# streets, more distinguishing context) do noticeably better -- see the
# per-generation_mode breakdown in eval/evaluate.py's output.


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)


def local_maxima(corr, num_peaks, suppress_radius, min_score):
    """Extract several distinct peaks from one correlation map via
    iterative max-find + local suppression."""
    peaks = []
    work = corr.copy()
    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val < min_score:
            break
        peaks.append((max_val, max_loc))
        cv2.circle(work, max_loc, suppress_radius, -2.0, -1)
    return peaks


def subpixel_offset(corr, loc):
    """Parabolic interpolation of the correlation surface around an
    integer-pixel peak, independently in x/y. Standard sub-pixel
    template-matching refinement -- without this, error is floored at
    ~0.5-1px of quantization no matter how good the candidate ranking is,
    which is enough on its own to blow the <2px/<1px thresholds."""
    x0, y0 = loc
    h, w = corr.shape[:2]
    dx = dy = 0.0
    if 0 < x0 < w - 1:
        fm1, f0, fp1 = corr[y0, x0 - 1], corr[y0, x0], corr[y0, x0 + 1]
        denom = fm1 - 2 * f0 + fp1
        if abs(denom) > 1e-9:
            dx = float(np.clip(0.5 * (fm1 - fp1) / denom, -0.5, 0.5))
    if 0 < y0 < h - 1:
        fm1, f0, fp1 = corr[y0 - 1, x0], corr[y0, x0], corr[y0 + 1, x0]
        denom = fm1 - 2 * f0 + fp1
        if abs(denom) > 1e-9:
            dy = float(np.clip(0.5 * (fm1 - fp1) / denom, -0.5, 0.5))
    return dx, dy


def generate_candidates(reference, search, top_k=50):
    """Returns list of (score, cx, cy, scale, angle), sorted best-first,
    deduplicated across the whole candidate pool (not just per-map)."""
    ref_h, ref_w = reference.shape[:2]
    all_candidates = []

    for scale in np.linspace(SCALE_MIN, SCALE_MAX, SCALE_STEPS):
        new_w = max(8, int(round(ref_w / scale)))
        new_h = max(8, int(round(ref_h / scale)))
        if new_w >= search.shape[1] or new_h >= search.shape[0]:
            continue
        resized_ref = cv2.resize(reference, (new_w, new_h), interpolation=cv2.INTER_AREA)

        for angle in np.linspace(ROT_MIN, ROT_MAX, ROT_STEPS):
            rotated = rotate_image(resized_ref, angle) if angle != 0 else resized_ref
            result = cv2.matchTemplate(search, rotated, cv2.TM_CCOEFF_NORMED)

            for score, loc in local_maxima(result, PEAKS_PER_MAP, PEAK_SUPPRESS_RADIUS, MIN_SCORE):
                dx, dy = subpixel_offset(result, loc)
                cx = loc[0] + dx + new_w / 2.0
                cy = loc[1] + dy + new_h / 2.0
                all_candidates.append((score, cx, cy, scale, angle))

    # global dedup: keep highest-scoring candidate within any cluster of nearby points
    all_candidates.sort(key=lambda c: -c[0])
    kept = []
    for cand in all_candidates:
        score, cx, cy, scale, angle = cand
        too_close = False
        for k in kept:
            if (cx - k[1]) ** 2 + (cy - k[2]) ** 2 < PEAK_SUPPRESS_RADIUS ** 2:
                too_close = True
                break
        if not too_close:
            kept.append(cand)
        if len(kept) >= top_k:
            break

    return kept
