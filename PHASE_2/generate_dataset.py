"""
================================================================================
 Multi-Region Die-Layout SEM Reference/Search BATCH Generator (v8 - Drift-Sense
 Phase 2: "Registration under Unknown Pose")
================================================================================
Standalone dataset-generator script for the Applied Materials "Drift-Sense"
hackathon problem statement. Generates paired 1000x1000 reference (unknown-
zoom close-up) / search (unknown-zoom wide-field) SEM-style images with
recorded ground-truth pose (x, y, theta, scale) and per-pair metadata.

Usage:
    python generate_dataset.py --architecture dram --num_pairs 70 --set A --output_dir ./out_setA
    python generate_dataset.py --architecture dram --num_pairs 70 --set B --output_dir ./out_setB
    python generate_dataset.py --architecture dram --num_pairs 40 --set C --output_dir ./out_setC
    python generate_dataset.py --architecture dram --num_pairs 20 --set D --output_dir ./out_setD
    python generate_dataset.py --architecture dram --num_pairs 20 --set E --output_dir ./out_setE
    python generate_dataset.py --architecture dram --num_pairs 200 --set mix --output_dir ./out_mix
    python generate_dataset.py --help

Required CLI parameters (per the hackathon deliverables table):
  --architecture   dram | finfet | mix   (which structure family to generate)
  --num_pairs      int                   (number of pairs to generate, >=30 recommended)
  --output_dir     path                  (directory the dataset is written into)

v8 changes vs v7 -- Phase 2 addendum ("Registration under Unknown Pose"):

  Everything in v7 is unchanged unless noted below. Three things the Phase 2
  addendum disclosed as changing are implemented here, plus the supporting
  infrastructure to validate against them locally:

  1. UNKNOWN ZOOM RATIO -- was a fixed 10x (SEARCH_NM_PER_PX = 10.0). Now
     drawn uniformly per-pair from --scale-min/--scale-max (default 8/12),
     via sample_zoom_ratio(). pick_search_window() now takes this per-pair
     ratio instead of the old global constant, and GT_scale in the ground-
     truth CSV records the exact value drawn -- your own train/val split can
     use this to fit a scale estimator or validate a search-over-scale
     matcher against ground truth.

  2. UNKNOWN ROTATION, +/-5 DEG -- was not modeled at all in v7 (reference
     and search were always extracted from the world with zero relative
     rotation). Now sample_rotation_deg() draws a per-pair angle from
     --rotation-max-deg (default 5), and rotate_search_image_and_point()
     rotates the *downsampled search image* about its own center (CCW
     positive, matching both cv2.getRotationMatrix2D's convention and the
     addendum's stated convention), forward-mapping the ground-truth center
     and bounding corners through the identical affine matrix so GT_X/GT_Y/
     GT_theta_deg stay pixel-exact after rotation. This intentionally
     mirrors matching.py's own rotate_image() (BORDER_REFLECT, about-center
     rotation) so a matcher's rotation-search assumptions are validated
     against a generator that rotates the same way.

  3. ABSENT PAIRS, ~20% BY DEFAULT -- was not modeled at all in v7 (the
     reference was always present). Now, --absent-fraction (default 0.20)
     of pairs get their search image cropped from build_absent_search_world()
     instead: a SECOND, independently-seeded world of the SAME architecture
     family (so it's "plausible and periodically similar" per the addendum's
     own wording) rather than a random unrelated image, guaranteeing the
     true reference pattern cannot appear in it by construction (different
     seed -> different per-mat pitch/params draws) while keeping the same
     general visual family. found=0, and GT_X/GT_Y/GT_theta_deg/GT_scale are
     written as 0, exactly matching the register.py output contract's rule
     ("When found=0, write 0 in the pose columns") -- so your own generated
     ground_truth.csv doubles as a template for what predictions.csv should
     look like.

  SUPPORTING ADDITIONS:
    - --degraded-fraction (default 0.50): of the PRESENT pairs, this
      fraction get pushed through a discrete --severity-levels (default 4)
      noise ladder (severity_to_noise_multiplier()) instead of the old
      continuous difficulty ramp, approximating the addendum's undisclosed
      "four undisclosed severity levels" for Set B. The remaining present
      pairs get mild, Set-A-like noise. This is a local, disclosed-category-
      only approximation of the organizer's held-out severity parameters --
      by design, per the addendum, you cannot know their exact levels; the
      point is to have *some* internal ladder to validate robustness
      against, not to reverse-engineer theirs.
    - "polygon scaling +/-20%" (named explicitly in the addendum's Set B
      description) is now actually wired up via the existing but previously
      unused apply_cd_bias_and_rounding() bias_px parameter (v7 always
      called it with bias_px=0.0 -- it only ever did rounding, never
      scaling). See build_noise_profile()'s new polygon_scale_pct_max arg.
    - --set {A,B,C,D,mix}: convenience presets matching the addendum's own
      Set A/B/C/D definitions (nominal / degraded / absent / optical-bonus),
      so you can generate each split separately and validate your matcher's
      A/B/C-weighted score (0.45*A + 0.55*B for localization, F1 on C for
      rejection) the same way the organizers will. --set D renders a best-
      effort pseudo-RGB "optical microscope analogue" (render_optical_rgb())
      -- this is NOT a validated optical simulator, just a 3-channel
      per-channel-jittered approximation, since Set D is bonus-only and the
      addendum gives no further detail on the real optical noise model.

  Everything else (citation registry, pattern-generator families, twin
  duplication, world/street layout, folder layout, PNG/metadata/CSV
  schema for existing fields) is UNCHANGED from v7 unless a field is called
  out as added/renamed above.

v10 changes vs v9 -- reference/ground-truth verification, ported from the
upstream Phase 2 reference pipeline (aayushraina21/drift-sense-synthetic-
data, src/phase2_pipeline.py + src/sem_imaging.py):

  1. GT POINT NOW TRACKED THROUGH *ALL* GEOMETRIC STEPS, NOT JUST ROTATION.
     v9 forward-mapped GT_X/GT_Y/GT_bbox_corners through the post-render
     rotation only. But render_image() also applies barrel distortion
     (apply_geometric_distortion, always) and raster drift (apply_raster_
     drift, search images only) to the pixels themselves -- both of those
     run BEFORE the rotation step, and both displace features relative to
     the pre-distortion coordinate the old code was mapping from. Any
     sample with barrel_k != 0 or shear/jitter != 0 (i.e. most degraded
     Set B pairs) therefore had GT_X/GT_Y off by however many pixels those
     two steps moved the true feature -- a silent few-px label error that
     got worse with severity. render_image() now optionally returns the
     exact row_shift array and barrel_k it used (see apply_raster_drift/
     apply_geometric_distortion below), and generate_sample() pushes the
     pre-rotation GT point through barrel_forward_pt() then
     drift_forward_pt() (closed-form inverses of the two backward/remap-
     based distortions) before the existing rotation step, so GT_X/GT_Y/
     GT_bbox_corners are pixel-exact in the final search.png regardless of
     degradation severity.

  2. REFERENCE-CROP UNIQUENESS VERIFICATION (periodic-pattern guard). DRAM/
     FinFET mats are periodic by construction, so a reference crop taken
     from deep inside a uniform mat can legitimately template-match BETTER
     somewhere else in the search image than at its own true, recorded
     location -- making that pair's label unreproducible by any correct
     matcher, no matter how good. verify_gt_unique() (a global cv2.
     matchTemplate correlation search against the true-pose template, with
     the winning peak's margin over the best competing peak outside a
     local exclusion window) now checks this for every present pair.
     pick_reference_crop() is retried (up to --max-crop-attempts times,
     default 6) until a candidate clears --verify-good-margin, falling
     back to the best on-label attempt if none does (only warning, never
     raising, so a long batch run never dies mid-way over one hard mat).
     Off by default: --no-verify-crops restores the old single-shot v9
     behavior (faster, no correlation search, no uniqueness guarantee).
     verify_ok / verify_err_px / verify_peak / verify_margin / verify_
     attempts are recorded per-pair in metadata/ground_truth.csv as a data-
     quality diagnostic (mirrors the upstream jury manifest's verify_*
     columns), not something register.py or a matcher ever sees.

  A/B/C/D set composition, folder layout, CLI contract, and every other
  v9 field are UNCHANGED.

v11 changes vs v10 -- local bonus Set E (legacy Set-D rendering):

  Adds a fifth, LOCAL-ONLY convenience preset, --set E. Set E now reuses the
  original Set-D implementation from generate_dataset.py: it builds the same
  grayscale DRAM/FinFET world used by Sets A/B/C, runs the normal SEM/noise/
  rotation/ground-truth pipeline, and then applies the legacy pseudo-RGB
  render_optical_rgb() conversion to the rendered reference and search images.
  The current Set-D implementation in this file is left unchanged.

v7 changes vs v6:

  - Added --architecture / --num_pairs / --output_dir / --seed as CLI
    arguments (argparse) instead of hardcoded globals, so the script can be
    run non-interactively and reproducibly as required by the hackathon
    deliverables table ("Must accept parameters: architecture style
    (DRAM/FinFET), number of pairs to generate, output directory").
  - --architecture dram/finfet now strictly restrict every generated pair
    to that family's pattern functions only (no cross-contamination).
    --architecture mix keeps the original v6 behaviour (DRAM + FinFET +
    the extra bonus pattern families: mesh_grid, ring_array,
    contact_array, layered_cell, logic_stripes, beol_interconnect,
    standard_cell_regular). Those bonus families are intentionally NOT
    available under pure "dram"/"finfet" runs, since the problem statement
    only asks for DRAM-style OR FinFET-style structures.
  - Each sample's metadata.json / ground_truth.csv row now also records
    "architecture_requested" for traceability.

v6 changes vs v5:

  - "edge_brightening" post-process step added to the noise pipeline,
    applied to BOTH reference and search images (independently -- each
    image gets its own random edge-brightening strength/kernel draw, on
    top of already-independent dose/beam/drift noise draws). This directly
    matches the sample-prompt PDF's explicit request: "a bit of
    edge-brightening to mimic how real SEM images show brighter contrast
    along feature edges." It is implemented as a Sobel-gradient-magnitude
    mask added back onto the image (classic SEM edge-contrast look: edges
    of raised/etched features scatter more secondary electrons and appear
    brighter), NOT a generic sharpen filter.

  - "world_id" now stored in metadata/CSV (a hash of the underlying
    world's random seed + generation mode + regions), so that a
    train/val/test split can group samples by underlying pattern world and
    avoid leakage between near-duplicate crops of the same world -- called
    out explicitly in the review notes above.

  - Everything else (citation registry, 5-level difficulty ladder, twin
    duplication, all pattern-style families, noise pipeline, GT format,
    folder layout) is UNCHANGED from v5.
================================================================================
"""

import os
import argparse
import shutil
import json
import random
import hashlib
import pickle
import numpy as np
import cv2
import pandas as pd

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ============================================================================
# -1. LITERATURE / CITATION REGISTRY
# ============================================================================

CITATIONS = {
    "irds2017":     "IRDS 2017 More Moore roadmap - https://irds.ieee.org/images/files/pdf/2017/2017/IRDS%20MM.pdf",
    "irds2024":     "IRDS 2024 More Moore roadmap - https://irds.ieee.org/images/files/pdf/2024/2024IRDS%20MM.pdf",
    "itrs2015":     "ITRS 2015 More Moore roadmap - https://www.semiconductors.org/wp-content/uploads/2018/06/5_2015-ITRS-2.0_More-Moore.pdf",
    "ibm_finfet14": "IBM Research, 'Opportunities and Challenges of FinFET as a Device Structure Candidate for 14nm Node CMOS Technology' - https://research.ibm.com/publications/opportunities-and-challenges-of-finfet-as-a-device-structure-candidate-for-14nm-node-cmos-technology",
    "semieng_7nm":  "Semiconductor Engineering, '7nm Fab Challenges' - https://semiengineering.com/7nm-fab-challenges/",
    "freepdk15":    "FreePDK15 predictive PDK paper - https://arxiv.org/pdf/2009.04600",
    "arxiv_ncfinfet": "arXiv 2007.14448 (NC-FinFET / IRDS last-FinFET-node context) - https://arxiv.org/pdf/2007.14448",
    "ti_wavy_bitline": "TI patent EP0780901A2 (arcuate moats / wavy bitlines) - https://patents.google.com/patent/EP0780901A2/en",
    "hynix_dram":   "EE Times, 'Hynix DRAM layout, process integration adapt to change' - https://www.eetimes.com/hynix-dram-layout-process-integration-adapt-to-change/",
    "sram_6t":      "US Patent 5,554,874 (6T SRAM cell) - https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5554874",
    "std_cell":     "US Patent 6,938,226 (7-track standard cell library) - https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6938226",
    "imec_logic":   "imec, 'View on logic technology roadmap' - https://www.imec-int.com/en/articles/view-logic-technology-roadmap",
    "imec_damascene": "imec, semi-damascene interconnect announcement - https://www.imec-int.com/en/articles/imec-demonstrates-semi-damascene-interconnects-fully-self-aligned-vias-18nm-metal-pitch",
    "ibm_beol":     "IBM Research BEOL blog (IEDM Cu interconnects) - https://research.ibm.com/blog/beol-cu-interconnects-iedm",
}

STYLE_CITATION_MAP = {
    "legacy_dram_1x":        (["hynix_dram"], "Regular repeating memory-cell dot array, motivated by DRAM layout regularity."),
    "legacy_finfet":         (["ibm_finfet14", "arxiv_ncfinfet"], "Parallel fin lines + crossing gate, motivated by FinFET device geometry."),
    "dram_staggered_realistic": (["hynix_dram", "sram_6t"], "Staggered repeated storage-node array, motivated by DRAM/SRAM repeated-cell geometry."),
    "finfet_via_realistic":  (["ibm_finfet14", "arxiv_ncfinfet", "irds2024"], "Fin-line grid with contact/via bands, motivated by FinFET + advanced-node interconnect context."),
    "wavy_dram_bitline":     (["ti_wavy_bitline", "hynix_dram"], "Non-perfectly-straight (arcuate) repeated bitline/dot geometry, motivated by patent literature on curved memory-array structures."),
    "mesh_grid":             (["imec_damascene", "ibm_beol"], "Dense orthogonal line + via grid, motivated by interconnect mesh density context."),
    "ring_array":            (["irds2017", "irds2024"], "Dense repeated ring/contact geometry, motivated by scaling-driven pattern density."),
    "contact_array_square":  (["imec_logic", "irds2017"], "Regular contact/via array, motivated by advanced-node contact density."),
    "contact_array_round":   (["imec_logic", "irds2017"], "Regular contact/via array, motivated by advanced-node contact density."),
    "layered_cell":          (["hynix_dram"], "Word-line/bit-line/storage-contact layered stack, motivated by DRAM cell process integration."),
    "logic_stripes":         (["imec_logic", "std_cell"], "Alternating-width stripe stack, motivated by logic/standard-cell routing texture."),
    "beol_interconnect":     (["imec_damascene", "ibm_beol"], "Multi-layer metal/via BEOL stack, motivated by semi-damascene and Cu interconnect process descriptions."),
    "standard_cell_regular": (["std_cell", "imec_logic"], "Fixed-pitch cell tracks bounded by row/cell-boundary lines, motivated by standard-cell library regularity."),
}

DIFFICULTY_LEVEL_5_BOUNDS = [
    (0.00, 0.20, 1, "easy"),
    (0.20, 0.40, 2, "moderate"),
    (0.40, 0.60, 3, "hard"),
    (0.60, 0.80, 4, "very_hard"),
    (0.80, 1.01, 5, "failure_case"),
]


def difficulty_level_5(difficulty):
    for lo, hi, level, name in DIFFICULTY_LEVEL_5_BOUNDS:
        if lo <= difficulty < hi:
            return level, name
    return 5, "failure_case"


# ============================================================================
# 0. GLOBAL SCALE / LAYOUT CONSTANTS
# ============================================================================

REF_PX = 1000
SEARCH_PX = 1000
REF_NM_PER_PX = 1.0
SEARCH_NM_PER_PX = 10.0
SCALE_RATIO = SEARCH_NM_PER_PX / REF_NM_PER_PX

MAT_SIZE_NM = 3000  # was 2600 (Phase 1, fixed 10x) -- bumped so the search-crop
                    # window still has jitter room at zoom_ratio close to Phase 2's
                    # scale_max=12 (see pick_search_window: at the old 2600, WORLD_PX
                    # (12000) == SEARCH_PX*scale_max (12000) exactly, so valid_range's
                    # hi collapsed to 0 and search_x0/search_y0 were forced to 0 for
                    # every high-zoom sample -- systematically less diverse than
                    # low-zoom samples. At 3000, WORLD_PX=13600 leaves ~1600px of
                    # slack at scale_max=12.
STREET_WIDTH_NM = 320
GRID_N = 4

WORLD_NM = GRID_N * MAT_SIZE_NM + (GRID_N + 1) * STREET_WIDTH_NM
WORLD_PX = int(WORLD_NM / REF_NM_PER_PX)

STRADDLE_PROB = 0.30
STREET_GRAY = 60

ANCHOR_PROB_LEGACY = 0.0
ANCHOR_PROB_REALISTIC = 0.15
ANCHOR_PROB_EXPANDED = 0.15
ANCHOR_PROB_SINGLE_FIELD = 0.30
REF_ON_ANCHOR_PROB = 0.25

BASE_SEED = 99991001

# Defaults used only if the script is run without CLI args (kept for
# backward-compatible import/testing). Real runs should pass --num_pairs,
# --output_dir and --architecture explicitly (see argparse in main()).
DEFAULT_N_SAMPLES = 500
DEFAULT_OUT_ROOT = "./synthetic_sem_dataset_holdout"
DEFAULT_ARCHITECTURE = "mix"

# ---------------------------------------------------------------------------
# Phase 2 ("Registration under Unknown Pose") defaults -- these mirror the
# disclosed bounds from the addendum. It is fine (and expected) to hard-code
# these bounds; only the exact per-pair value is unknown at inference time.
# ---------------------------------------------------------------------------
DEFAULT_SCALE_MIN = 8.0             # disclosed lower bound of the zoom ratio
DEFAULT_SCALE_MAX = 12.0            # disclosed upper bound of the zoom ratio
DEFAULT_ROTATION_MAX_DEG = 5.0      # disclosed +/- bound on rotation
DEFAULT_ABSENT_FRACTION = 0.20      # ~Set C's share of the addendum's 180
DEFAULT_DEGRADED_FRACTION = 0.50    # ~ Set B's share of the PRESENT pairs
DEFAULT_SEVERITY_LEVELS = 4         # addendum: "four undisclosed severity levels"
DEFAULT_POLYGON_SCALE_PCT_MAX = 20.0  # addendum: "polygon scaling +/-20%"

# ---------------------------------------------------------------------------
# v10: reference/ground-truth verification (see section 5 below and the
# module docstring addendum at the bottom of this constants block for why
# these exist). Ported from the upstream Phase 2 reference pipeline
# (aayushraina21/drift-sense-synthetic-data, src/phase2_pipeline.py's
# verify_gt_unique / drift_forward_pt / barrel_forward_pt), adapted to this
# script's world/street/mat model and noise-profile field names instead of
# the upstream's single fixed-canvas + Phase2Params model.
# ---------------------------------------------------------------------------
DEFAULT_VERIFY_CROPS = True
DEFAULT_MAX_CROP_ATTEMPTS = 6
DEFAULT_VERIFY_GOOD_MARGIN = 0.10
DEFAULT_VERIFY_MIN_MARGIN = 0.02
DEFAULT_VERIFY_TOL_PX = 3.0

SET_PRESETS = {
    # (absent_fraction, degraded_fraction, optical, hint)
    "A":   (0.0, 0.0, False, False),  # nominal, present, full scale/rotation range
    "B":   (0.0, 1.0, False, False),  # degraded, present, full scale/rotation range
    "C":   (1.0, 0.0, False, False),  # absent -- correct answer is found=0
    "D":   (0.0, 0.0, True,  False),  # optical bonus, present, full native-color RGB analogue
    "E":   (0.0, 0.0, False, True),   # local bonus: uses the legacy Set-D pseudo-RGB rendering path
    "mix": None,                      # use whatever --absent-fraction / --degraded-fraction were passed
}


def _grid_coords(size, pitch, phase=0):
    return np.arange(pitch // 2 + phase, size, pitch)


# ============================================================================
# 1a. LEGACY PATTERN FAMILY
# ============================================================================

def legacy_dram_1x(size):
    pitch = random.randint(45, 95)
    dot_r = max(3, pitch // 6)
    img = np.zeros((size, size), dtype=np.uint8)
    for y in _grid_coords(size, pitch):
        for x in _grid_coords(size, pitch):
            cv2.circle(img, (int(x), int(y)), dot_r, 255, -1)
    return img, {"style": "legacy_dram_1x", "pitch_nm": pitch, "dot_r_nm": dot_r}


def legacy_finfet(size):
    pitch_fin = random.randint(22, 55)
    lw_fin = max(3, pitch_fin // 4)
    pitch_gate = random.randint(160, 420)
    lw_gate = random.randint(20, 55)
    img = np.zeros((size, size), dtype=np.uint8)
    cols = np.arange(size)
    fin_mask = (cols % pitch_fin) < lw_fin
    img[:, fin_mask] = 190
    rows = np.arange(size)
    gate_mask = (rows % pitch_gate) < lw_gate
    img[gate_mask, :] = 255
    return img, {"style": "legacy_finfet", "pitch_fin_nm": pitch_fin, "lw_fin_nm": lw_fin,
                 "pitch_gate_nm": pitch_gate, "lw_gate_nm": lw_gate}


# ============================================================================
# 1b. REALISTIC DRAM / FINFET FAMILY
# ============================================================================

def realistic_dram_staggered(size):
    pitch = random.randint(60, 120)
    cap_len = random.randint(int(pitch * 0.7), int(pitch * 1.1))
    cap_w = max(6, pitch // 6)
    dot_r = max(4, pitch // 8)
    bg = random.randint(55, 80)
    img = np.full((size, size), bg, dtype=np.uint8)

    row = 0
    y = pitch // 2
    while y < size:
        offset = (pitch // 2) if (row % 2 == 1) else 0
        x = pitch // 2 + offset
        while x < size:
            axes = (cap_len // 2, cap_w // 2)
            cv2.ellipse(img, (int(x), int(y)), axes, 45, 0, 360, 205, -1)
            ex = int(x - (cap_len / 2) * 0.70)
            ey = int(y + (cap_len / 2) * 0.70)
            cv2.circle(img, (ex, ey), dot_r, 225, -1)
            x += pitch
        y += pitch
        row += 1

    return img, {"style": "dram_staggered_realistic", "pitch_nm": pitch,
                 "cap_len_nm": cap_len, "cap_w_nm": cap_w, "dot_r_nm": dot_r}


def realistic_finfet_via(size):
    pitch_v = random.randint(50, 110)
    lw_v = max(6, pitch_v // 4)
    band_h = random.randint(60, 140)
    bg = random.randint(45, 65)
    img = np.full((size, size), bg, dtype=np.uint8)

    for x in _grid_coords(size, pitch_v):
        x0, x1 = int(x - lw_v / 2), int(x + lw_v / 2)
        img[:, max(0, x0):min(size, x1)] = 190

    n_bands = random.randint(2, 4)
    for b in range(n_bands):
        by = int(size * (b + 1) / (n_bands + 1))
        img[max(0, by - band_h // 2):min(size, by + band_h // 2), :] = 20

    return img, {"style": "finfet_via_realistic", "pitch_v_nm": pitch_v,
                 "lw_v_nm": lw_v, "band_h_nm": band_h}


def stamp_realistic_via(pattern, cx, cy):
    shape = random.choice(["cross", "square"])
    s = random.randint(60, 120)
    gray_val = random.choice(list(range(190, 251, 15)) + list(range(5, 46, 15)))
    cx, cy = int(cx), int(cy)

    if shape == "cross":
        arm = s
        thick = max(10, s // 4)
        cv2.rectangle(pattern, (cx - thick // 2, cy - arm), (cx + thick // 2, cy + arm), gray_val, -1)
        cv2.rectangle(pattern, (cx - arm, cy - thick // 2), (cx + arm, cy + thick // 2), gray_val, -1)
    else:
        cv2.rectangle(pattern, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), gray_val, -1)
        notch = s // 3
        cv2.rectangle(pattern, (cx + s // 2 - notch, cy + s // 2 - notch),
                      (cx + s // 2 + 4, cy + s // 2 + 4), STREET_GRAY, -1)

    return {"anchor_shape": shape, "anchor_size_nm": s, "anchor_x": cx, "anchor_y": cy}


# ============================================================================
# 1b-2. LITERATURE-MOTIVATED ADDITIONS (v5, retained in v6)
# ============================================================================

def wavy_dram_bitline(size):
    pitch = random.randint(55, 100)
    dot_r = max(4, pitch // 7)
    bg = random.randint(50, 70)
    img = np.full((size, size), bg, dtype=np.uint8)

    for y in _grid_coords(size, pitch):
        for x in _grid_coords(size, pitch):
            cv2.circle(img, (int(x), int(y)), dot_r, 205, -1)

    curvature_strength = random.uniform(0.01, 0.05)
    n_lines = size // pitch
    for i in range(n_lines):
        y0 = int(pitch // 2 + i * pitch)
        pts = []
        for x in range(0, size, 10):
            y = y0 + int(curvature_strength * pitch * np.sin(2 * np.pi * x / (size / 3)))
            pts.append((x, y))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p1, p2, 150, max(2, dot_r // 2))

    map_x, map_y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    map_y = map_y + (curvature_strength * pitch * np.sin(2 * np.pi * map_x / (size / 3))).astype(np.float32)
    img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return img, {"style": "wavy_dram_bitline", "pitch_nm": pitch, "dot_r_nm": dot_r,
                 "curvature_strength": round(curvature_strength, 4)}


def beol_interconnect(size):
    pitch_m1 = random.randint(30, 70)
    lw_m1 = max(4, pitch_m1 // 3)
    pitch_m2 = random.randint(40, 90)
    lw_m2 = max(4, pitch_m2 // 3)
    via_r = random.randint(4, 10)
    bg = random.randint(20, 35)
    img = np.full((size, size), bg, dtype=np.uint8)

    for y in _grid_coords(size, pitch_m1):
        y0, y1 = int(y - lw_m1 / 2), int(y + lw_m1 / 2)
        img[max(0, y0):min(size, y1), :] = 130
    for x in _grid_coords(size, pitch_m2):
        x0, x1 = int(x - lw_m2 / 2), int(x + lw_m2 / 2)
        img[:, max(0, x0):min(size, x1)] = np.maximum(img[:, max(0, x0):min(size, x1)], 170)
    for y in _grid_coords(size, pitch_m1):
        for x in _grid_coords(size, pitch_m2):
            if random.random() < 0.6:
                cv2.circle(img, (int(x), int(y)), via_r, 235, -1)

    return img, {"style": "beol_interconnect", "pitch_m1_nm": pitch_m1, "pitch_m2_nm": pitch_m2,
                 "lw_m1_nm": lw_m1, "lw_m2_nm": lw_m2, "via_r_nm": via_r}


def standard_cell_regular(size):
    track_pitch = random.randint(18, 40)
    track_w = max(3, track_pitch // 3)
    row_height = random.randint(140, 260)
    bg = random.randint(60, 85)
    img = np.full((size, size), bg, dtype=np.uint8)

    for x in _grid_coords(size, track_pitch):
        x0, x1 = int(x - track_w / 2), int(x + track_w / 2)
        img[:, max(0, x0):min(size, x1)] = 175

    for y in _grid_coords(size, row_height):
        y0, y1 = int(y - 4), int(y + 4)
        img[max(0, y0):min(size, y1), :] = 30

    return img, {"style": "standard_cell_regular", "track_pitch_nm": track_pitch,
                 "track_w_nm": track_w, "row_height_nm": row_height}


# ============================================================================
# 1c. EXPANDED PATTERN FAMILY
# ============================================================================

def build_mesh_grid(size, pitch, line_w, via_size, via_prob=0.5):
    img = np.full((size, size), 65, dtype=np.uint8)
    for x in _grid_coords(size, pitch):
        x0, x1 = int(x - line_w / 2), int(x + line_w / 2)
        img[:, max(0, x0):min(size, x1)] = 160
    for y in _grid_coords(size, pitch):
        y0, y1 = int(y - line_w / 2), int(y + line_w / 2)
        img[max(0, y0):min(size, y1), :] = 160
    for y in _grid_coords(size, pitch):
        for x in _grid_coords(size, pitch):
            if random.random() < via_prob:
                half = via_size // 2
                cv2.rectangle(img, (int(x - half), int(y - half)),
                              (int(x + half), int(y + half)), 225, -1)
    return img, {"style": "mesh_grid", "pitch_nm": pitch, "line_w_nm": line_w,
                 "via_size_nm": via_size}


def build_ring_array(size, pitch, r_outer, r_inner):
    img = np.full((size, size), 45, dtype=np.uint8)
    for y in _grid_coords(size, pitch):
        for x in _grid_coords(size, pitch):
            cv2.circle(img, (int(x), int(y)), r_outer, 200, -1)
            cv2.circle(img, (int(x), int(y)), r_inner, 45, -1)
    return img, {"style": "ring_array", "pitch_nm": pitch, "r_outer_nm": r_outer,
                 "r_inner_nm": r_inner}


def build_contact_array(size, pitch, contact_size, shape):
    img = np.full((size, size), 90, dtype=np.uint8)
    half = contact_size // 2
    for y in _grid_coords(size, pitch):
        for x in _grid_coords(size, pitch):
            if shape == "square":
                cv2.rectangle(img, (int(x - half), int(y - half)),
                              (int(x + half), int(y + half)), 220, -1)
            else:
                cv2.circle(img, (int(x), int(y)), half, 220, -1)
    return img, {"style": f"contact_array_{shape}", "pitch_nm": pitch,
                 "contact_size_nm": contact_size}


def build_layered_cell(size, pitch_word, lw_word, pitch_bit, lw_bit, dot_r):
    img = np.full((size, size), 25, dtype=np.uint8)
    rows = np.arange(size)
    word_mask = (rows % pitch_word) < lw_word
    img[word_mask, :] = 110
    cols = np.arange(size)
    bit_mask = (cols % pitch_bit) < lw_bit
    img[:, bit_mask] = np.maximum(img[:, bit_mask], 165)
    row = 0
    for y in _grid_coords(size, pitch_word):
        offset = (pitch_bit // 2) if (row % 2 == 1) else 0
        for x in _grid_coords(size, pitch_bit, phase=offset):
            cv2.circle(img, (int(x), int(y)), dot_r, 235, -1)
        row += 1
    return img, {"style": "layered_cell", "pitch_word_nm": pitch_word,
                 "lw_word_nm": lw_word, "pitch_bit_nm": pitch_bit,
                 "lw_bit_nm": lw_bit, "dot_r_nm": dot_r}


def build_logic_stripes(size, min_w, max_w, palette):
    img = np.zeros((size, size), dtype=np.uint8)
    x = 0
    while x < size:
        w = random.randint(min_w, max_w)
        val = random.choice(palette)
        img[:, x:min(size, x + w)] = val
        x += w
    return img, {"style": "logic_stripes", "min_w_nm": min_w, "max_w_nm": max_w}


def expanded_random(size):
    style = random.choice(["mesh_grid", "ring_array", "contact_array", "layered_cell",
                            "logic_stripes", "beol_interconnect", "standard_cell_regular"])
    if style == "beol_interconnect":
        return beol_interconnect(size)
    if style == "standard_cell_regular":
        return standard_cell_regular(size)
    if style == "mesh_grid":
        pitch = random.randint(90, 220)
        return build_mesh_grid(size, pitch, random.randint(6, 18), random.randint(14, 34))
    if style == "ring_array":
        pitch = random.randint(90, 200)
        r_outer = random.randint(20, 40)
        r_inner = max(6, r_outer - random.randint(8, 16))
        return build_ring_array(size, pitch, r_outer, r_inner)
    if style == "contact_array":
        pitch = random.randint(60, 160)
        return build_contact_array(size, pitch, random.randint(14, 40),
                                    random.choice(["square", "round"]))
    if style == "layered_cell":
        pw = random.randint(50, 110)
        pb = random.randint(50, 110)
        return build_layered_cell(size, pw, max(8, pw // 3), pb, max(8, pb // 3),
                                   random.randint(6, 16))
    palette = random.sample(range(30, 230, 20), k=5)
    return build_logic_stripes(size, random.randint(20, 45), random.randint(50, 140), palette)


# ============================================================================
# 1d. ARCHITECTURE-AWARE STYLE POOLS  (Drift-Sense requirement: "Must accept
# parameters: architecture style (DRAM/FinFET), number of pairs to generate,
# output directory.")
#
# --architecture dram    -> EVERY generated pair uses ONLY DRAM-style
#                            pattern functions (legacy_dram_1x,
#                            realistic_dram_staggered, wavy_dram_bitline).
# --architecture finfet  -> EVERY generated pair uses ONLY FinFET-style
#                            pattern functions (legacy_finfet,
#                            realistic_finfet_via).
# --architecture mix     -> original v6 behaviour: DRAM + FinFET + the
#                            extra "expanded" pattern families (mesh_grid,
#                            ring_array, contact_array, layered_cell,
#                            logic_stripes, beol_interconnect,
#                            standard_cell_regular). The problem statement
#                            only asks for DRAM-style OR FinFET-style
#                            structures, so those extra bonus families are
#                            kept ONLY under "mix" and never appear in a
#                            pure "dram" or "finfet" run.
# ============================================================================

DRAM_STYLE_FNS = [legacy_dram_1x, realistic_dram_staggered, wavy_dram_bitline]
FINFET_STYLE_FNS = [legacy_finfet, realistic_finfet_via]

# Generation-mode weights per architecture. "expanded" is only ever
# sampled when architecture == "mix".
GEN_MODE_WEIGHTS_BY_ARCH = {
    "dram": {
        "legacy": 0.35,
        "realistic": 0.45,
        "single_field": 0.20,
    },
    "finfet": {
        "legacy": 0.35,
        "realistic": 0.45,
        "single_field": 0.20,
    },
    "mix": {
        "legacy": 0.25,
        "realistic": 0.35,
        "expanded": 0.25,
        "single_field": 0.15,
    },
}


def choose_generation_mode(architecture):
    """Pick a generation mode ('legacy' / 'realistic' / 'expanded' /
    'single_field') using the weight table for the requested architecture.
    'expanded' never appears unless architecture == 'mix'."""
    weights_dict = GEN_MODE_WEIGHTS_BY_ARCH[architecture]
    modes, weights = zip(*weights_dict.items())
    return random.choices(modes, weights=weights, k=1)[0]


def get_family_style_fns(family, architecture):
    """Return the list of pattern-generator functions usable for a given
    ('legacy' / 'realistic' / 'expanded') family, filtered to the requested
    architecture. This is the single choke point that guarantees
    --architecture dram never emits a FinFET pattern and vice versa."""
    if family == "legacy":
        if architecture == "dram":
            return [legacy_dram_1x]
        if architecture == "finfet":
            return [legacy_finfet]
        return [legacy_dram_1x, legacy_finfet]  # mix

    if family == "realistic":
        if architecture == "dram":
            return [realistic_dram_staggered, wavy_dram_bitline]
        if architecture == "finfet":
            return [realistic_finfet_via]
        return [realistic_dram_staggered, realistic_finfet_via, wavy_dram_bitline]  # mix

    # family == "expanded" is only ever reached when architecture == "mix"
    # (see GEN_MODE_WEIGHTS_BY_ARCH), but keep this safe regardless.
    return [expanded_random]


def get_single_field_style_fns(architecture):
    """Style pool used by build_world_single_field() for the requested
    architecture."""
    if architecture == "dram":
        return DRAM_STYLE_FNS
    if architecture == "finfet":
        return FINFET_STYLE_FNS
    return [legacy_dram_1x, legacy_finfet, realistic_dram_staggered,
            realistic_finfet_via, expanded_random]  # mix


# ============================================================================
# 1e. PHASE 2 -- NATIVE-COLOR PATTERN FAMILY (Set D / optical bonus, v9)
#
# v8 produced Set D by taking an already-rendered GRAYSCALE image and
# post-processing it into pseudo-RGB (per-channel gain/blur/noise --
# render_optical_rgb(), now removed). That's gone: Set D images are now
# drawn directly in color from the start, using distinct hues per
# structural layer (word_line/bit_line/storage_contact etc.), then run
# through the IDENTICAL noise/rotation/absent-pair pipeline used for
# Sets A/B/C (verified to work unmodified on 3-channel arrays -- Gaussian
# blur, dilate/erode, warpAffine, Sobel, and remap all operate per-channel).
#
# Colors are BGR tuples (cv2.imwrite/cv2 drawing convention).
# ============================================================================

def color_dram_layered(size):
    """word_line / bit_line / storage_contact layered DRAM-cell view --
    horizontal blue word-lines, vertical green bit-lines, orange-red
    storage-contact dots at their intersections, black substrate."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    pitch_word = random.randint(60, 110)
    lw_word = max(8, int(pitch_word * 0.55))
    pitch_bit = random.randint(70, 130)
    lw_bit = max(8, int(pitch_bit * 0.35))
    dot_r = max(6, min(pitch_word, pitch_bit) // 6)

    word_color = (190, 90, 30)     # blue-ish bar
    bit_color = (60, 165, 70)      # green-ish bar
    contact_color = (35, 75, 220)  # orange-red dot

    for y in _grid_coords(size, pitch_word):
        y0, y1 = int(y - lw_word / 2), int(y + lw_word / 2)
        img[max(0, y0):min(size, y1), :] = word_color
    for x in _grid_coords(size, pitch_bit):
        x0, x1 = int(x - lw_bit / 2), int(x + lw_bit / 2)
        img[:, max(0, x0):min(size, x1)] = bit_color
    for y in _grid_coords(size, pitch_word):
        for x in _grid_coords(size, pitch_bit):
            cv2.circle(img, (int(x), int(y)), dot_r, contact_color, -1)

    return img, {"style": "color_dram_layered", "pitch_word_nm": pitch_word,
                 "pitch_bit_nm": pitch_bit, "dot_r_nm": dot_r}


def color_hex_via(size):
    """Hexagonal via/contact grid: orange/tan field, teal-green hex outline
    lattice, magenta contact-dot clusters inside alternating cells."""
    img = np.full((size, size, 3), (60, 140, 210), dtype=np.uint8)  # tan/orange bg
    hex_r = random.randint(45, 75)
    hex_color = (110, 150, 70)
    dot_color = (150, 40, 200)
    dx = hex_r * 1.5
    dy = hex_r * np.sqrt(3)
    row = 0
    y = hex_r
    while y < size + hex_r:
        offset = (dx / 2) if (row % 2 == 1) else 0
        x = hex_r + offset
        while x < size + hex_r:
            pts = np.array([
                [x + hex_r * np.cos(np.pi / 3 * k), y + hex_r * np.sin(np.pi / 3 * k)]
                for k in range(6)
            ], dtype=np.int32)
            cv2.polylines(img, [pts], True, hex_color, max(2, hex_r // 20))
            if random.random() < 0.7:
                n_dots = random.randint(3, 6)
                for _ in range(n_dots):
                    ddx = random.uniform(-hex_r * 0.5, hex_r * 0.5)
                    ddy = random.uniform(-hex_r * 0.5, hex_r * 0.5)
                    cv2.circle(img, (int(x + ddx), int(y + ddy)), max(3, hex_r // 10), dot_color, -1)
            x += dx
        y += dy
        row += 1

    return img, {"style": "color_hex_via", "hex_r_nm": hex_r}


def color_plaid_scatter(size):
    """Plaid weave (blue verticals over dark-orange horizontals on tan bg)
    with sparse white contact dots scattered off-grid."""
    bg = (70, 130, 190)
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    pitch_v = random.randint(35, 65)
    pitch_h = random.randint(35, 65)
    v_color = (150, 70, 30)
    h_color = (30, 70, 120)
    for x in _grid_coords(size, pitch_v):
        x0, x1 = int(x - 3), int(x + 3)
        img[:, max(0, x0):min(size, x1)] = v_color
    for y in _grid_coords(size, pitch_h):
        y0, y1 = int(y - 3), int(y + 3)
        img[max(0, y0):min(size, y1), :] = h_color

    n_dots = random.randint(15, 30)
    for _ in range(n_dots):
        cx, cy = random.randint(0, size - 1), random.randint(0, size - 1)
        cv2.circle(img, (cx, cy), random.randint(6, 12), (235, 235, 235), -1)

    return img, {"style": "color_plaid_scatter", "pitch_v_nm": pitch_v, "pitch_h_nm": pitch_h}


def color_teal_gold_grid(size):
    """Gold vertical lines + teal horizontal track bars on a light
    lavender-gray field, cyan dot markers at intersections."""
    img = np.full((size, size, 3), (210, 200, 195), dtype=np.uint8)
    pitch_v = random.randint(40, 70)
    pitch_h = random.randint(60, 100)
    gold = (30, 190, 210)
    teal = (150, 130, 20)
    cyan = (210, 210, 60)
    for x in _grid_coords(size, pitch_v):
        img[:, max(0, int(x) - 3):min(size, int(x) + 3)] = gold
    for y in _grid_coords(size, pitch_h):
        y0, y1 = int(y - 10), int(y + 10)
        img[max(0, y0):min(size, y1), :] = teal
    for y in _grid_coords(size, pitch_h):
        for x in _grid_coords(size, pitch_v):
            cv2.circle(img, (int(x), int(y)), 6, cyan, -1)

    return img, {"style": "color_teal_gold_grid", "pitch_v_nm": pitch_v, "pitch_h_nm": pitch_h}


def color_mint_coral_wavy(size):
    """Black field, mint-green dot grid, coral wavy horizontal lines
    threading through it."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    pitch = random.randint(45, 80)
    mint = (170, 220, 150)
    coral = (110, 130, 240)
    for y in _grid_coords(size, pitch):
        for x in _grid_coords(size, pitch):
            cv2.circle(img, (int(x), int(y)), max(3, pitch // 10), mint, -1)

    curvature = random.uniform(0.02, 0.06)
    n_lines = max(3, size // (pitch * 2))
    for i in range(n_lines):
        y0 = int(pitch + i * pitch * 2)
        pts = []
        for x in range(0, size, 8):
            y = y0 + int(curvature * pitch * 6 * np.sin(2 * np.pi * x / (size / 2.5)))
            pts.append((x, y))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p1, p2, coral, 4)

    return img, {"style": "color_mint_coral_wavy", "pitch_nm": pitch}


def color_magenta_dash_scatter(size):
    """Black field, sparse dashed magenta cross-hatch, small teal dot
    scatter -- a low-density interconnect-probe-style look."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    magenta = (200, 30, 200)
    teal_c = (170, 160, 40)
    n_v = random.randint(2, 4)
    n_h = random.randint(2, 4)
    for _ in range(n_v):
        x = random.randint(0, size - 1)
        y = 0
        while y < size:
            seg = random.randint(20, 50)
            if random.random() < 0.6:
                cv2.line(img, (x, y), (x, min(size, y + seg)), magenta, 3)
            y += seg + random.randint(10, 30)
    for _ in range(n_h):
        y = random.randint(0, size - 1)
        x = 0
        while x < size:
            seg = random.randint(20, 50)
            if random.random() < 0.6:
                cv2.line(img, (x, y), (min(size, x + seg), y), magenta, 3)
            x += seg + random.randint(10, 30)

    n_dots = random.randint(80, 160)
    for _ in range(n_dots):
        cx, cy = random.randint(0, size - 1), random.randint(0, size - 1)
        cv2.circle(img, (cx, cy), random.randint(3, 6), teal_c, -1)

    return img, {"style": "color_magenta_dash_scatter"}


def color_olive_fin_field(size):
    """FinFET color analogue: olive-green field, pale vertical fin
    stripes, red wavy horizontal interconnect lines, blue via/contact
    dot clusters scattered along the fins."""
    img = np.full((size, size, 3), (55, 95, 75), dtype=np.uint8)  # olive
    pitch_fin = random.randint(50, 90)
    lw_fin = max(6, pitch_fin // 5)
    fin_color = (170, 190, 190)
    red = (50, 60, 200)
    blue = (200, 110, 60)

    for x in _grid_coords(size, pitch_fin):
        x0, x1 = int(x - lw_fin / 2), int(x + lw_fin / 2)
        img[:, max(0, x0):min(size, x1)] = fin_color

    n_lines = random.randint(4, 7)
    curvature = random.uniform(0.02, 0.05)
    for i in range(n_lines):
        y0 = int(size * (i + 1) / (n_lines + 1))
        pts = []
        for x in range(0, size, 8):
            y = y0 + int(curvature * 120 * np.sin(2 * np.pi * x / (size / 3)))
            pts.append((x, y))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(img, p1, p2, red, 3)

    n_clusters = random.randint(3, 6)
    for _ in range(n_clusters):
        ccx, ccy = random.randint(0, size - 1), random.randint(0, size - 1)
        for _ in range(random.randint(4, 9)):
            dx = random.randint(-40, 40)
            dy = random.randint(-40, 40)
            cv2.circle(img, (ccx + dx, ccy + dy), random.randint(4, 8), blue, -1)

    return img, {"style": "color_olive_fin_field", "pitch_fin_nm": pitch_fin}


def color_purple_yellow_dash(size):
    """Black field, dashed purple verticals + dashed yellow horizontals
    forming a sparse grid, small yellow-green dots scattered at random."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    purple = (190, 60, 130)
    yellow = (40, 210, 220)
    yellow_green = (60, 210, 170)

    pitch_v = random.randint(70, 130)
    pitch_h = random.randint(80, 140)
    for x in _grid_coords(size, pitch_v):
        y = 0
        while y < size:
            seg = random.randint(15, 35)
            cv2.line(img, (int(x), y), (int(x), min(size, y + seg)), purple, 3)
            y += seg + random.randint(8, 20)
    for y in _grid_coords(size, pitch_h):
        x = 0
        while x < size:
            seg = random.randint(15, 35)
            cv2.line(img, (x, int(y)), (min(size, x + seg), int(y)), yellow, 3)
            x += seg + random.randint(8, 20)

    n_dots = random.randint(20, 40)
    for _ in range(n_dots):
        cx, cy = random.randint(0, size - 1), random.randint(0, size - 1)
        cv2.circle(img, (cx, cy), random.randint(4, 7), yellow_green, -1)

    return img, {"style": "color_purple_yellow_dash", "pitch_v_nm": pitch_v, "pitch_h_nm": pitch_h}


COLOR_STYLE_FNS_DRAM = [color_dram_layered, color_hex_via, color_plaid_scatter,
                        color_teal_gold_grid, color_magenta_dash_scatter]
COLOR_STYLE_FNS_FINFET = [color_olive_fin_field, color_mint_coral_wavy, color_purple_yellow_dash]


def get_color_style_fns(architecture):
    """Style pool used for Set D (optical/RGB) world building -- mirrors
    get_family_style_fns()/get_single_field_style_fns() but for the native-
    color pattern family (see module docstring section 1e)."""
    if architecture == "dram":
        return COLOR_STYLE_FNS_DRAM
    if architecture == "finfet":
        return COLOR_STYLE_FNS_FINFET
    return COLOR_STYLE_FNS_DRAM + COLOR_STYLE_FNS_FINFET  # mix


# ============================================================================
# 1f. PHASE 2 -- COLOR-HINT PATTERN FAMILY (Set E / accent-color bonus, v11)
#
# Set D (section 1e) is fully native color -- every pixel is drawn from a
# distinct-hue palette from the start. Set E is the opposite kind of bonus:
# it reuses the IDENTICAL grayscale DRAM/FinFET pattern generators used by
# Sets A/B/C, converts each rendered mat to a 3-channel BGR canvas (still
# visually pure grayscale, R==G==B everywhere), then finds THAT PATTERN'S
# OWN shape edges/outlines (dot rims, cap/ellipse boundaries, fin-line and
# gate-bar edges, etc. -- via Canny + findContours on the gray render
# itself) and re-draws a random sparse subset of just those outlines in
# color, alpha-blended against the local gray value under each one. So the
# color always sits ON the pattern's own polygon/line boundaries -- never
# unrelated shapes scattered on top -- which is what makes it read as "a
# few of this structure's own edges happen to be picked out in color"
# not a colorized image. Wired into the SAME style_fns_override/is_color
# choke point build_world_grid()/build_world_single_field() already expose
# for Set D, so the rest of the pipeline (noise, blur, rotation, absent-pair
# logic, GT bbox) is untouched and already verified to work on 3-channel
# arrays.
# ============================================================================

# BGR tuples (cv2 convention), reused sparingly -- these are the same family
# of saturated accent hues Set D uses for its structural layers, just applied
# as small sparse accents here instead of covering the whole canvas.
_HINT_ACCENT_PALETTE = [
    (35, 75, 220),    # orange-red
    (190, 90, 30),    # blue
    (60, 165, 70),    # green
    (150, 40, 200),   # magenta
    (30, 190, 210),   # gold
    (200, 110, 60),   # steel blue
    (110, 130, 240),  # coral
]


def add_color_hints(gray_img, size, alpha=0.85, contour_frac_range=(0.015, 0.035)):
    """Takes a single-channel grayscale pattern (as produced by any of the
    normal DRAM/FinFET style functions) and returns a 3-channel BGR image
    that is almost entirely grayscale, with a SPARSE SUBSET of the pattern's
    own shape edges/outlines -- the actual dot/cap/fin/bar boundaries already
    present in the image -- lightly tinted in color.

    This does NOT paint unrelated color shapes on top of the pattern. It
    finds the real contours of the existing structures (cv2.Canny +
    cv2.findContours on the gray pattern itself), keeps only a random
    fraction of them, and re-draws just those outlines in a hint color,
    alpha-blended against the local gray value under each one. The result
    reads as "a few of this SEM pattern's own polygon/line edges happen to
    be picked out in color", not confetti scattered over the image.
    """
    out = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)  # uint8, cheap even at world scale

    edges = cv2.Canny(gray_img, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_perim = max(12.0, size * 0.003)
    contours = [c for c in contours if cv2.arcLength(c, False) >= min_perim]
    if not contours:
        return out

    frac = random.uniform(*contour_frac_range)
    n_pick = max(1, int(round(len(contours) * frac)))
    chosen = random.sample(contours, min(n_pick, len(contours)))

    thickness = max(2, int(round(size * 0.0025)))

    def _blend(color, gray_val):
        return tuple(int(round(alpha * c + (1.0 - alpha) * gray_val)) for c in color)

    for c in chosen:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cx, cy = int(c[0][0][0]), int(c[0][0][1])
        cx = int(np.clip(cx, 0, size - 1))
        cy = int(np.clip(cy, 0, size - 1))
        gray_val = int(gray_img[cy, cx])
        color = _blend(random.choice(_HINT_ACCENT_PALETTE), gray_val)
        cv2.drawContours(out, [c], -1, color, thickness, lineType=cv2.LINE_AA)

    return out


def make_hint_style_fn(base_style_fn):
    """Wraps a normal grayscale style_fn(size) -> (img_2d, meta) into a
    Set-E style_fn(size) -> (img_3ch, meta) that runs the same pattern
    through add_color_hints(). Keeps the same signature/contract the world
    builders expect from any entry in style_fns_override."""
    def _hint_fn(size):
        gray, meta = base_style_fn(size)
        img = add_color_hints(gray, size)
        meta = dict(meta)
        meta["style"] = f"{meta.get('style', base_style_fn.__name__)}_hint"
        return img, meta
    _hint_fn.__name__ = f"hint_{base_style_fn.__name__}"
    return _hint_fn


HINT_STYLE_FNS_DRAM = [make_hint_style_fn(fn) for fn in DRAM_STYLE_FNS]
HINT_STYLE_FNS_FINFET = [make_hint_style_fn(fn) for fn in FINFET_STYLE_FNS]


def get_hint_style_fns(architecture):
    """Style pool used for Set E (color-hint bonus) world building -- mirrors
    get_color_style_fns() but wraps the grayscale DRAM/FinFET generators
    instead of using a separate native-color pattern family (see section 1f)."""
    if architecture == "dram":
        return HINT_STYLE_FNS_DRAM
    if architecture == "finfet":
        return HINT_STYLE_FNS_FINFET
    return HINT_STYLE_FNS_DRAM + HINT_STYLE_FNS_FINFET  # mix


# ============================================================================
# 2. STREETS / SCRIBE-LINE LADDER MARKS (grid modes only)
# ============================================================================

def draw_ladder_anchors(canvas, street_x0, street_width, world_size):
    rung_height = random.randint(70, 140)
    rung_gap = random.randint(50, 100)
    y = 0
    toggle = True
    while y < world_size:
        color = 200 if toggle else 25
        y2 = min(world_size, y + rung_height)
        canvas[y:y2, street_x0:street_x0 + street_width] = color
        toggle = not toggle
        y += rung_height + rung_gap


# ============================================================================
# 3. WORLD BUILDERS
# ============================================================================

def build_world_grid(family, difficulty=0.5, architecture="mix", style_fns_override=None, is_color=False):
    style_fns = style_fns_override if style_fns_override is not None else get_family_style_fns(family, architecture)
    if family == "legacy":
        anchor_prob = ANCHOR_PROB_LEGACY
    elif family == "realistic":
        anchor_prob = ANCHOR_PROB_REALISTIC
    else:
        anchor_prob = ANCHOR_PROB_EXPANDED
    if is_color:
        # anchor stamping (stamp_realistic_via) uses scalar cv2 draw colors
        # that only fill channel 0 on a multi-channel canvas -- rather than
        # rewrite that path for color, Set D simply skips anchors (a minor,
        # cosmetic-only feature; not needed for the bonus set's purpose).
        anchor_prob = 0.0

    canvas_shape = (WORLD_PX, WORLD_PX, 3) if is_color else (WORLD_PX, WORLD_PX)
    canvas = np.full(canvas_shape, STREET_GRAY, dtype=np.uint8)
    mat_px = int(MAT_SIZE_NM / REF_NM_PER_PX)
    street_px = int(STREET_WIDTH_NM / REF_NM_PER_PX)
    positions = [street_px + i * (mat_px + street_px) for i in range(GRID_N)]

    duplicate_prob = lerp(0.0, 0.75, max(0.0, (difficulty - 0.4) / 0.6))
    twin_pattern, twin_params, twin_mat_indices = None, None, set()
    twin_style_fn = None
    n_mats = GRID_N * GRID_N
    if random.random() < duplicate_prob:
        twin_style_fn = random.choice(style_fns)
        twin_pattern, twin_params = twin_style_fn(mat_px)
        n_twins = random.randint(2, 4)
        twin_mat_indices = set(random.sample(range(n_mats), min(n_twins, n_mats)))

    regions = []
    mat_i = 0
    for row in range(GRID_N):
        for col in range(GRID_N):
            y0, x0 = positions[row], positions[col]

            if mat_i in twin_mat_indices:
                pattern, params = twin_pattern.copy(), dict(twin_params)
                params["is_twin_duplicate"] = True
                params["style_fn"] = twin_style_fn
            else:
                style_fn = random.choice(style_fns)
                pattern, params = style_fn(mat_px)
                params["is_twin_duplicate"] = False
                params["style_fn"] = style_fn

            has_anchor = (not params["is_twin_duplicate"]) and (random.random() < anchor_prob)
            anchor_info = None
            if has_anchor:
                margin = 220
                acx = random.randint(margin, mat_px - margin)
                acy = random.randint(margin, mat_px - margin)
                anchor_info = stamp_realistic_via(pattern, acx, acy)
                anchor_info["anchor_x_world"] = x0 + anchor_info["anchor_x"]
                anchor_info["anchor_y_world"] = y0 + anchor_info["anchor_y"]

            canvas[y0:y0 + mat_px, x0:x0 + mat_px] = pattern
            regions.append({"row": row, "col": col, "x0": x0, "y0": y0, "size": mat_px,
                             "has_anchor": has_anchor, "anchor": anchor_info, **params})
            mat_i += 1

    street_x_positions = [i * (mat_px + street_px) for i in range(GRID_N + 1)]
    for sx in street_x_positions:
        draw_ladder_anchors(canvas, sx, street_px, WORLD_PX)

    return canvas, regions, street_x_positions, mat_px, street_px


def build_world_single_field(architecture="mix", style_fns_override=None, is_color=False):
    style_fns = style_fns_override if style_fns_override is not None else get_single_field_style_fns(architecture)
    style_fn = random.choice(style_fns)
    canvas, params = style_fn(WORLD_PX)
    params["style_fn"] = style_fn

    has_anchor = (not is_color) and (random.random() < ANCHOR_PROB_SINGLE_FIELD)
    anchor_info = None
    if has_anchor:
        margin = 600
        acx = random.randint(margin, WORLD_PX - margin)
        acy = random.randint(margin, WORLD_PX - margin)
        anchor_info = stamp_realistic_via(canvas, acx, acy)
        anchor_info["anchor_x_world"] = acx
        anchor_info["anchor_y_world"] = acy

    regions = [{"row": 0, "col": 0, "x0": 0, "y0": 0, "size": WORLD_PX,
                "has_anchor": has_anchor, "anchor": anchor_info, **params}]
    return canvas, regions, [], WORLD_PX, 0


# ============================================================================
# 4. PHYSICS / SEM IMAGING NOISE PIPELINE
# ============================================================================

def apply_geometric_distortion(img, k):
    if k == 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y, x = np.indices((h, w), dtype=np.float32)
    xn, yn = (x - cx) / cx, (y - cy) / cy
    r2 = xn ** 2 + yn ** 2
    factor = 1.0 + k * r2
    map_x = (xn * factor) * cx + cx
    map_y = (yn * factor) * cy + cy
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def barrel_forward_pt(x, y, k, w, h):
    """Forward barrel map for a single point -- the closed-form inverse of
    apply_geometric_distortion()'s backward/remap formulation.

    apply_geometric_distortion is a backward map: it samples
    output(x, y) <- input(map_x, map_y) where map = center + (xn*factor)*c,
    factor = 1 + k*r_out^2 (normalized radius). So a feature that truly
    sits at input radius r_in ends up, after the forward warp, at whatever
    output radius r_out satisfies r_out*(1 + k*r_out^2) = r_in. Solved here
    by Newton iteration (ported from upstream sem_imaging.py's identical
    backward-map convention). Non-square images use separate cx/cy but a
    shared normalized radius, matching apply_geometric_distortion above.
    """
    if k == 0.0:
        return x, y
    cx, cy = w / 2.0, h / 2.0
    nx, ny = (x - cx) / cx, (y - cy) / cy
    r_in = float(np.hypot(nx, ny))
    if r_in < 1e-9:
        return x, y
    r = r_in
    for _ in range(40):
        f = r * (1.0 + k * r * r) - r_in
        df = 1.0 + 3.0 * k * r * r
        step = f / df
        r -= step
        if abs(step) < 1e-12:
            break
    ratio = r / r_in
    return cx + nx * ratio * cx, cy + ny * ratio * cy


def apply_cd_bias_and_rounding(img, bias_px, round_sigma):
    if abs(bias_px) >= 0.5:
        k = max(1, int(round(abs(bias_px))) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img = cv2.dilate(img, kernel) if bias_px > 0 else cv2.erode(img, kernel)
    if round_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), round_sigma)
    return img


def apply_beam_blur(img, sigma_px, astig_ratio):
    sigma_x = max(sigma_px * astig_ratio, 0.3)
    sigma_y = max(sigma_px / astig_ratio, 0.3)
    kx = max(1, int(round(sigma_x * 4)) | 1)
    ky = max(1, int(round(sigma_y * 4)) | 1)
    out = cv2.GaussianBlur(img.astype(np.float32), (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_raster_drift(img, shear_px, jitter_px, return_shift=False):
    """Per-row horizontal raster-drift warp (backward map: output(x, y)
    samples input(x + row_shift[y], y)).

    return_shift=True also returns the exact row_shift array drawn, so the
    ground-truth point can be pushed through the identical per-row shift
    afterwards (see drift_forward_pt) instead of accumulating unexplained
    label error at high shear/jitter severity.
    """
    h, w = img.shape[:2]
    if shear_px == 0 and jitter_px == 0:
        row_shift = np.zeros(h, dtype=np.float32)
        return (img, row_shift) if return_shift else img
    rows = np.arange(h, dtype=np.float32)
    shear_shift = shear_px * (rows / h)
    jitter = np.random.normal(0, jitter_px, size=h).astype(np.float32)
    row_shift = shear_shift + jitter
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
    return (out, row_shift) if return_shift else out


def drift_forward_pt(x, y, row_shift):
    """Where a feature at input (x, y) ends up after apply_raster_drift.

    The warp is a backward map (see apply_raster_drift), so a feature at
    row y moves by -row_shift[y] in x. Ported from upstream phase2_pipeline
    .py's identical drift_forward_pt."""
    h = len(row_shift)
    yi = int(np.clip(round(y), 0, h - 1))
    return x - float(row_shift[yi]), y


def apply_dose_noise(img, dose, gaussian_sigma=3.0):
    img_f = img.astype(np.float32) / 255.0
    electrons = np.clip(img_f * dose, 0, None)
    noisy_electrons = np.random.poisson(electrons).astype(np.float32)
    out = noisy_electrons / dose * 255.0
    out += np.random.normal(0, gaussian_sigma, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_charging_streaks(img, intensity=40, frequency=0.05):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    n_streaks = max(1, int(h * frequency))
    rows = np.random.choice(h, size=n_streaks, replace=False)
    for r in rows:
        thickness = random.randint(1, 3)
        r2 = min(h, r + thickness)
        out[r:r2, :] += intensity
    return np.clip(out, 0, 255).astype(np.uint8)


def add_speckle_noise(img, sigma=0.15):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) * (1.0 + noise)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.01):
    out = img.copy()
    n = out.size
    n_each = int(amount * n * 0.5)
    ys = np.random.randint(0, out.shape[0], n_each)
    xs = np.random.randint(0, out.shape[1], n_each)
    out[ys, xs] = 255
    ys = np.random.randint(0, out.shape[0], n_each)
    xs = np.random.randint(0, out.shape[1], n_each)
    out[ys, xs] = 0
    return out


def add_edge_brightening(img, strength=0.35, blur_sigma=1.2):
    """SEM edge-contrast effect: edges of raised/etched features scatter
    more secondary electrons and read out brighter than flat regions.
    Implemented as a Sobel-gradient-magnitude mask, smoothed, then added
    back onto the image -- independent per call (independent random
    strength each time this is invoked), matching the reference/search
    images needing their OWN separate edge-brightening draw, not a shared
    one. This is distinct from the beam-blur / dose-noise steps: those
    degrade the whole image, this selectively lifts intensity right at
    feature boundaries.
    """
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = cv2.GaussianBlur(grad_mag, (0, 0), blur_sigma)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    out = img_f + strength * 255.0 * grad_mag
    return np.clip(out, 0, 255).astype(np.uint8)


def lerp(a, b, t):
    return a + (b - a) * t


def build_noise_profile(difficulty, is_search, polygon_scale_pct_max=0.0):
    """difficulty here is a 0..1 NOISE-SEVERITY fraction (for search images
    in Phase 2, this is severity_to_noise_multiplier()'s output for degraded
    pairs, or a small fixed mild value for nominal/Set-A-like pairs -- see
    generate_sample()), not necessarily the structural difficulty ramp used
    for twin-duplicate/anchor placement.

    polygon_scale_pct_max > 0 enables the addendum's 'polygon scaling
    +/-20%' degradation: draws a +/- pct linewidth bias (scaled by
    difficulty, so severity level 1 barely scales, level 4 scales close to
    the full +/-max), converted to an approximate dilation/erosion bias_px
    via apply_cd_bias_and_rounding()'s existing (previously unused) bias_px
    parameter -- v7 always called it with bias_px=0.0."""
    if is_search:
        dose = random.uniform(lerp(1.3, 0.12, difficulty), lerp(1.6, 0.30, difficulty))
        shear = random.uniform(lerp(0.1, 3.0, difficulty), lerp(0.4, 6.0, difficulty))
        jitter = random.uniform(lerp(0.0, 1.0, difficulty), lerp(0.2, 2.5, difficulty))
        beam_spot = random.uniform(lerp(4.0, 7.5, difficulty), lerp(5.5, 10.5, difficulty))
        astig = random.uniform(lerp(1.02, 1.35, difficulty), lerp(1.10, 1.70, difficulty))
        barrel_k = random.uniform(-0.002, 0.002) * difficulty
        p_active = lerp(0.10, 0.85, difficulty)
    else:
        dose = random.uniform(lerp(1.8, 0.9, difficulty), lerp(2.2, 1.3, difficulty))
        shear, jitter = 0.0, 0.0
        beam_spot = random.uniform(4.0, 5.5)
        astig = random.uniform(1.02, 1.15)
        barrel_k = 0.0
        p_active = lerp(0.03, 0.35, difficulty)

    active_types = []
    charging_on = random.random() < p_active
    speckle_on = random.random() < p_active
    saltpepper_on = random.random() < p_active
    if charging_on:
        active_types.append("charging")
    if speckle_on:
        active_types.append("speckle")
    if saltpepper_on:
        active_types.append("salt_pepper")
    if is_search and (shear > 0.5 or jitter > 0.3):
        active_types.append("raster_drift")
    active_types.append("shot_noise")
    active_types.append("edge_brightening")   # always applied, independently per image

    edge_strength = random.uniform(0.20, 0.55)
    edge_blur_sigma = random.uniform(0.8, 1.8)

    polygon_scale_pct = 0.0
    polygon_bias_px = 0.0
    if is_search and polygon_scale_pct_max > 0:
        polygon_scale_pct = random.uniform(-polygon_scale_pct_max, polygon_scale_pct_max) * difficulty
        # nominal ~40px feature linewidth heuristic (generator has many
        # families with different pitches; this is an approximation, not a
        # per-style-exact conversion) -- see apply_polygon_scaling docstring
        # in build_noise_profile's module docstring above.
        nominal_linewidth_px = 40.0
        polygon_bias_px = (polygon_scale_pct / 100.0) * nominal_linewidth_px * 0.5

    profile = {
        "dose_scale": dose,
        "shear_px": shear,
        "jitter_px": jitter,
        "beam_spot_nm": beam_spot,
        "astig_ratio": astig,
        "barrel_k": barrel_k,
        "charging_on": charging_on,
        "charging_intensity": random.uniform(lerp(10, 30, difficulty), lerp(25, 60, difficulty)) if charging_on else 0,
        "charging_freq": random.uniform(0.02, 0.10) if charging_on else 0,
        "speckle_on": speckle_on,
        "speckle_sigma": random.uniform(lerp(0.03, 0.12, difficulty), lerp(0.10, 0.28, difficulty)) if speckle_on else 0,
        "saltpepper_on": saltpepper_on,
        "saltpepper_amount": random.uniform(lerp(0.002, 0.008, difficulty), lerp(0.006, 0.035, difficulty)) if saltpepper_on else 0,
        "edge_strength": edge_strength,
        "edge_blur_sigma": edge_blur_sigma,
        "polygon_scale_pct": round(polygon_scale_pct, 3),
        "polygon_bias_px": round(polygon_bias_px, 3),
        "active_types": active_types,
    }
    if abs(polygon_scale_pct) > 1e-6:
        profile["active_types"].append("polygon_scaling")
    return profile


def render_image(world_crop, nm_per_px, base_dose, is_search, profile):
    """Renders world_crop through the full noise/distortion pipeline.

    Returns (img, geom) where geom = {"barrel_k": <float>, "row_shift":
    <1D array or None>} records the exact position-changing parameters
    used (barrel distortion is always applied; raster drift only for
    search images) -- generate_sample() pushes the ground-truth point
    through both (barrel_forward_pt then drift_forward_pt) before the
    later rotation step, so GT_X/GT_Y stay pixel-exact regardless of
    degradation severity (v10; see module docstring).
    """
    img = world_crop.copy()
    img = apply_geometric_distortion(img, profile["barrel_k"])

    round_sigma = max(0.3, 1.0 / nm_per_px)
    img = apply_cd_bias_and_rounding(img, profile.get("polygon_bias_px", 0.0), round_sigma)

    sigma_px = profile["beam_spot_nm"] / nm_per_px
    img = apply_beam_blur(img, sigma_px, profile["astig_ratio"])

    row_shift = None
    if is_search:
        img, row_shift = apply_raster_drift(img, shear_px=profile["shear_px"],
                                             jitter_px=profile["jitter_px"], return_shift=True)

    img = apply_dose_noise(img, base_dose * profile["dose_scale"])

    if profile["charging_on"]:
        img = add_charging_streaks(img, intensity=profile["charging_intensity"],
                                    frequency=profile["charging_freq"])
    if profile["speckle_on"]:
        img = add_speckle_noise(img, sigma=profile["speckle_sigma"])
    if profile["saltpepper_on"]:
        img = add_salt_pepper_noise(img, amount=profile["saltpepper_amount"])

    # edge-brightening applied LAST, after all other degradation, so the
    # brighter edge halo itself carries a touch of the same noise texture
    # (matches how real secondary-electron edge contrast looks noisy too)
    img = add_edge_brightening(img, strength=profile["edge_strength"],
                                blur_sigma=profile["edge_blur_sigma"])

    geom = {"barrel_k": profile["barrel_k"], "row_shift": row_shift}
    return img, geom


# ============================================================================
# 5. REFERENCE / SEARCH CROP STRATEGY
# ============================================================================

def pick_reference_crop(regions, street_x_positions, mat_px):
    ref_px = REF_PX
    allow_straddle = bool(street_x_positions) and len(street_x_positions) > 2

    if allow_straddle and random.random() < STRADDLE_PROB:
        internal_streets = street_x_positions[1:-1]
        sx = random.choice(internal_streets)
        street_px = int(STREET_WIDTH_NM / REF_NM_PER_PX)
        min_left_margin = 150
        max_left_margin = ref_px - street_px - 150
        left_margin = random.randint(min_left_margin, max(min_left_margin, max_left_margin))
        ref_x0 = int(np.clip(sx - left_margin, 0, WORLD_PX - ref_px))

        rows_present = sorted(set(r["row"] for r in regions))
        row = random.choice(rows_present)
        row_y0 = next(r["y0"] for r in regions if r["row"] == row)
        lo = max(0, row_y0 + 20)
        hi = min(WORLD_PX - ref_px, row_y0 + mat_px - ref_px - 20)
        ref_y0 = random.randint(lo, hi) if hi > lo else row_y0
        return ref_x0, ref_y0, True, False

    anchor_regions = [r for r in regions if r["has_anchor"]]
    use_anchor = anchor_regions and (random.random() < REF_ON_ANCHOR_PROB)

    if use_anchor:
        box = random.choice(anchor_regions)
        ax = box["anchor"]["anchor_x_world"]
        ay = box["anchor"]["anchor_y_world"]
        jitter = ref_px // 4
        cx = ax + random.randint(-jitter, jitter)
        cy = ay + random.randint(-jitter, jitter)
        ref_x0 = int(np.clip(cx - ref_px // 2, box["x0"] + 5, box["x0"] + box["size"] - ref_px - 5))
        ref_y0 = int(np.clip(cy - ref_px // 2, box["y0"] + 5, box["y0"] + box["size"] - ref_px - 5))
        return ref_x0, ref_y0, False, True

    box = random.choice(regions)
    lo_x, hi_x = box["x0"] + 10, box["x0"] + box["size"] - ref_px - 10
    lo_y, hi_y = box["y0"] + 10, box["y0"] + box["size"] - ref_px - 10
    ref_x0 = random.randint(lo_x, hi_x) if hi_x > lo_x else box["x0"]
    ref_y0 = random.randint(lo_y, hi_y) if hi_y > lo_y else box["y0"]
    return ref_x0, ref_y0, False, False


def _template_from_reference(ref_img, zoom_ratio, theta_deg):
    """Renders what the reference crop SHOULD look like inside the final
    search raster: blur to approximate the search-side beam PSF + z-fold
    downscale, rotate by the sample's own theta, and scale down by zoom_
    ratio. Ported from upstream phase2_pipeline.py's identical helper,
    used only by verify_gt_unique() below -- never written to disk."""
    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if ref_img.ndim == 3 else ref_img
    k = max(2, int(round(zoom_ratio)))
    blurred = cv2.blur(gray, (k, k))
    out = max(8, int(round(gray.shape[0] / zoom_ratio)))
    M = cv2.getRotationMatrix2D(((gray.shape[1] - 1) / 2, (gray.shape[0] - 1) / 2),
                                 theta_deg, 1.0 / zoom_ratio)
    M[0, 2] += (out - 1) / 2 - (gray.shape[1] - 1) / 2
    M[1, 2] += (out - 1) / 2 - (gray.shape[0] - 1) / 2
    return cv2.warpAffine(blurred, M, (out, out), flags=cv2.INTER_LINEAR)


def verify_gt_unique(reference_img, search_img, gt, tol_px=DEFAULT_VERIFY_TOL_PX):
    """Is the recorded (gt['x'], gt['y']) label actually where a correct
    matcher would find this reference in this search image?

    DRAM/FinFET mats are periodic, so a reference crop taken from deep
    inside a uniform mat can correlate better SOMEWHERE ELSE in the search
    image than at its own true, recorded location -- which would make the
    label unreproducible no matter how good the matching algorithm is.
    This runs a global cv2.matchTemplate correlation search at the known
    pose (zoom_ratio, theta_deg) and reports: whether the correlation peak
    actually lands on the recorded label (within tol_px), and the margin
    of that peak over the best competing peak outside a local exclusion
    window around it (a small/negative margin means the pattern repeats
    convincingly elsewhere -- an ambiguous, low-quality crop choice).
    Ported from upstream phase2_pipeline.py's identical verify_gt_unique.
    """
    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY) if search_img.ndim == 3 else search_img
    tpl = _template_from_reference(reference_img, gt["scale"], gt["theta"])
    if tpl.shape[0] >= search_gray.shape[0] or tpl.shape[1] >= search_gray.shape[1]:
        return {"ok": False, "err_px": float("nan"), "peak": float("nan"),
                "second_peak": float("nan"), "margin": float("-inf")}
    res = cv2.matchTemplate(search_gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    half = (tpl.shape[0] - 1) / 2.0
    px, py = loc[0] + half, loc[1] + half
    err = float(np.hypot(px - gt["x"], py - gt["y"]))

    masked = res.copy()
    r_excl = int(max(tpl.shape[0] * 0.6, 12))
    x0, y0 = max(loc[0] - r_excl, 0), max(loc[1] - r_excl, 0)
    masked[y0:loc[1] + r_excl, x0:loc[0] + r_excl] = -1.0
    _, second, _, _ = cv2.minMaxLoc(masked)

    return {"ok": err <= tol_px, "err_px": err, "peak": float(peak),
            "second_peak": float(second), "margin": float(peak - second)}


def pick_search_window(ref_x0, ref_y0, zoom_ratio):
    """zoom_ratio replaces the old fixed SEARCH_NM_PER_PX (Phase 2: unknown,
    uniform in [scale_min, scale_max] per pair -- see sample_zoom_ratio())."""
    search_px = int(round((SEARCH_PX * zoom_ratio) / REF_NM_PER_PX))
    search_px = min(search_px, WORLD_PX)  # guard the ratio=scale_max edge case
    ref_px = REF_PX

    def valid_range(ref_o):
        lo = max(0, ref_o + ref_px - search_px)
        hi = min(ref_o, WORLD_PX - search_px)
        if hi < lo:
            hi = lo
        return lo, hi

    lo_x, hi_x = valid_range(ref_x0)
    lo_y, hi_y = valid_range(ref_y0)
    search_x0 = random.randint(lo_x, hi_x)
    search_y0 = random.randint(lo_y, hi_y)
    return search_x0, search_y0, search_px


# ============================================================================
# 5b. PHASE 2 -- SCALE / ROTATION / ABSENT-PAIR / SEVERITY HELPERS
# ============================================================================

def sample_zoom_ratio(scale_min, scale_max):
    """Uniform per-pair zoom ratio, replacing the old fixed SEARCH_NM_PER_PX."""
    return random.uniform(scale_min, scale_max)


def sample_rotation_deg(rotation_max_deg):
    """Uniform per-pair rotation in [-rotation_max_deg, +rotation_max_deg],
    CCW positive (see rotate_search_image_and_point)."""
    if rotation_max_deg <= 0:
        return 0.0
    return random.uniform(-rotation_max_deg, rotation_max_deg)


def rotate_search_image_and_point(img, theta_deg, points_xy):
    """Rotate `img` about its own center by theta_deg and forward-map each
    (x, y) in points_xy through the identical affine matrix.

    Convention: cv2.getRotationMatrix2D uses positive angle = counter-
    clockwise, which matches the Phase 2 addendum's stated convention for
    theta ("CCW positive, measured about the match centre"). This also
    matches matching.py's own rotate_image() helper, so the generator and
    the reference matcher agree on which way "positive rotation" turns the
    image -- important, because a sign mismatch here would silently make
    every rotated ground-truth pair wrong by 2*theta.

    BORDER_REFLECT (not black-fill) is used so a matcher never gets a free
    "corner is pure black" cue -- same choice matching.py already makes.
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    mapped = []
    for (x, y) in points_xy:
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        mapped.append((nx, ny))
    return rotated, mapped


def severity_to_noise_multiplier(level, n_levels):
    """Map a discrete severity level (1..n_levels) to a continuous noise
    multiplier in (0, 1], reusing build_noise_profile()'s existing
    difficulty-indexed lerp() ramps instead of a parallel noise system.
    Levels are jittered within their band (not fixed constants) since the
    organizer's real ladder is undisclosed by design -- this is a local
    stand-in to stress-test against, not an attempt to guess theirs."""
    n_levels = max(1, n_levels)
    band = 1.0 / n_levels
    lo = (level - 1) * band
    hi = min(1.0, level * band)
    return random.uniform(lo, max(lo, hi))


def build_absent_search_world(unique_seed, gen_mode, architecture, forced_style_fn=None, size=None, is_color=False):
    """Builds the source canvas for an absent (Set C style) search crop.

    If `forced_style_fn` is given (the SAME generator function that produced
    the reference's own region -- see generate_sample()), this renders
    directly from that function under an independent RNG draw: same pattern
    *family* by construction (guaranteed visual/periodic similarity, not
    left to chance), but a fresh pitch/phase/duty-cycle draw, so the true
    reference instance cannot appear in it. This is both a better proxy for
    the addendum's 'plausible and periodically similar' wording AND far
    cheaper than the old approach (no world/streets/mats built at all --
    just style_fn(size) once).

    Falls back to the old behavior (fresh, unrelated second world of the
    same architecture, different seed) when forced_style_fn is None -- kept
    for diversity / as a sanity check that style-matching is actually doing
    something (compare with --no-absent-style-match). is_color routes this
    fallback to the color style pool (Set D) instead of grayscale -- in
    practice Set D always has absent_fraction=0 (see SET_PRESETS), so this
    only matters for unusual manual --optical + --absent-fraction combos.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        absent_seed = (unique_seed * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
        random.seed(absent_seed)
        np.random.seed(absent_seed % (2**32 - 1))

        if forced_style_fn is not None:
            render_size = size if size is not None else WORLD_PX
            canvas, _ = forced_style_fn(render_size)
        elif gen_mode == "single_field":
            color_pool = get_color_style_fns(architecture) if is_color else None
            canvas, regions_b, streets_b, mat_px_b, street_px_b = build_world_single_field(
                architecture, style_fns_override=color_pool, is_color=is_color)
        else:
            color_pool = get_color_style_fns(architecture) if is_color else None
            canvas, regions_b, streets_b, mat_px_b, street_px_b = build_world_grid(
                gen_mode, difficulty=random.random(), architecture=architecture,
                style_fns_override=color_pool, is_color=is_color)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return canvas


def find_containing_style_fn(regions, x0, y0, ref_px):
    """Finds the region whose bounding box contains the reference crop's
    center point, and returns its style_fn (the exact pattern generator
    that produced it) -- used to build a style-matched absent pair. Falls
    back to None (caller uses the old random-world fallback) if the center
    happens to land in street background (rare) or region tracking is
    missing for any reason."""
    cx, cy = x0 + ref_px / 2.0, y0 + ref_px / 2.0
    for r in regions:
        if r["x0"] <= cx < r["x0"] + r["size"] and r["y0"] <= cy < r["y0"] + r["size"]:
            return r.get("style_fn")
    return None


# ============================================================================
# 5b. LEGACY SET-D PSEUDO-RGB RENDERING (reused by local Set E)
# ============================================================================
def render_optical_rgb(gray_img):
    """Best-effort pseudo-RGB 'optical microscope analogue' from the original
    Set-D implementation in generate_dataset.py."""
    h, w = gray_img.shape[:2]
    channel_gains = (random.uniform(0.92, 1.05),   # B
                     random.uniform(0.97, 1.05),   # G
                     random.uniform(0.90, 1.02))   # R
    channel_blur = (random.uniform(0.3, 0.9),
                    random.uniform(0.2, 0.6),
                    random.uniform(0.4, 1.0))
    channels = []
    base = gray_img.astype(np.float32)
    for gain, blur_sigma in zip(channel_gains, channel_blur):
        ch = base * gain
        if blur_sigma > 0:
            ch = cv2.GaussianBlur(ch, (0, 0), blur_sigma)
        ch += np.random.normal(0, 2.5, ch.shape).astype(np.float32)
        channels.append(np.clip(ch, 0, 255).astype(np.uint8))
    return cv2.merge(channels)  # BGR order for cv2.imwrite


# ============================================================================
# 6. PER-SAMPLE ORCHESTRATOR
# ============================================================================

# (choose_generation_mode() lives up in section 1d, since it needs to be
# architecture-aware and the architecture-filtered style pools are defined
# there right after the pattern functions.)

def compute_world_id(gen_mode, unique_seed, regions):
    """Stable hash identifying the underlying pattern 'world' this sample was
    cropped from -- used to group samples for train/val/test splitting so
    near-duplicate crops of the SAME world never end up split across sets
    (leakage risk flagged in the review notes)."""
    style_sig = "|".join(sorted(r["style"] for r in regions))
    raw = f"{gen_mode}:{unique_seed}:{style_sig}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _finite_round(v, nd=3):
    """round(), but returns None for NaN/+-inf instead of writing them
    literally into the CSV (pandas would render them as empty/±inf strings
    that downstream code has to special-case)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def generate_sample(unique_seed, difficulty, architecture="mix",
                     scale_min=DEFAULT_SCALE_MIN, scale_max=DEFAULT_SCALE_MAX,
                     rotation_max_deg=DEFAULT_ROTATION_MAX_DEG,
                     absent_fraction=DEFAULT_ABSENT_FRACTION,
                     degraded_fraction=DEFAULT_DEGRADED_FRACTION,
                     severity_levels=DEFAULT_SEVERITY_LEVELS,
                     polygon_scale_pct_max=DEFAULT_POLYGON_SCALE_PCT_MAX,
                     optical=False, hint=False, absent_style_match=True,
                     verify_crops=DEFAULT_VERIFY_CROPS,
                     max_crop_attempts=DEFAULT_MAX_CROP_ATTEMPTS,
                     verify_good_margin=DEFAULT_VERIFY_GOOD_MARGIN,
                     verify_min_margin=DEFAULT_VERIFY_MIN_MARGIN,
                     verify_tol_px=DEFAULT_VERIFY_TOL_PX):
    random.seed(unique_seed)
    np.random.seed(unique_seed % (2**32 - 1))

    gen_mode = choose_generation_mode(architecture)
    if optical:
        # Set D (v9): build the world directly in native color using the
        # color pattern family (section 1e) instead of grayscale + post-hoc
        # tinting -- everything downstream (crop, noise, rotation, absent-
        # pair logic, GT bbox) is untouched and already verified to work on
        # 3-channel arrays.
        color_pool = get_color_style_fns(architecture)
        if gen_mode == "single_field":
            world, regions, street_x_positions, mat_px, street_px = build_world_single_field(
                architecture, style_fns_override=color_pool, is_color=True)
        else:
            world, regions, street_x_positions, mat_px, street_px = build_world_grid(
                gen_mode, difficulty=difficulty, architecture=architecture,
                style_fns_override=color_pool, is_color=True)
    elif hint:
        # Set E now uses the legacy Set-D implementation from generate_dataset.py:
        # build the normal grayscale DRAM/FinFET world first, then convert the
        # rendered reference/search images to pseudo-RGB after all SEM geometry
        # and pose operations. This replaces the former v11 color-hint family.
        if gen_mode == "single_field":
            world, regions, street_x_positions, mat_px, street_px = build_world_single_field(architecture)
        else:
            world, regions, street_x_positions, mat_px, street_px = build_world_grid(
                gen_mode, difficulty=difficulty, architecture=architecture)
    elif gen_mode == "single_field":
        world, regions, street_x_positions, mat_px, street_px = build_world_single_field(architecture)
    else:
        world, regions, street_x_positions, mat_px, street_px = build_world_grid(
            gen_mode, difficulty=difficulty, architecture=architecture)

    world_id = compute_world_id(gen_mode, unique_seed, regions)

    # ---- Phase 2: per-pair pose (unknown to the matcher, known here) ----
    zoom_ratio = sample_zoom_ratio(scale_min, scale_max)
    theta_deg = sample_rotation_deg(rotation_max_deg)

    is_absent = random.random() < absent_fraction
    is_degraded = (not is_absent) and (random.random() < degraded_fraction)
    if is_degraded:
        severity_level = random.randint(1, severity_levels)
        noise_severity = severity_to_noise_multiplier(severity_level, severity_levels)
    else:
        severity_level = 0
        noise_severity = random.uniform(0.05, 0.25)  # mild, Set-A-like

    if is_absent:
        # Set C: search image is rendered from the SAME style generator
        # function that produced the reference's own region (style-matched,
        # see find_containing_style_fn/build_absent_search_world), under an
        # independent RNG draw -- guarantees visual/periodic-family
        # similarity by construction while the true reference instance
        # cannot appear in it (fresh pitch/phase/duty-cycle draw). No true
        # instance exists, so there is nothing to verify -- single shot.
        ref_x0, ref_y0, straddled, on_anchor = pick_reference_crop(regions, street_x_positions, mat_px)
        ref_world_crop = world[ref_y0:ref_y0 + REF_PX, ref_x0:ref_x0 + REF_PX]
        ref_profile = build_noise_profile(difficulty, is_search=False)
        reference_img, _ = render_image(ref_world_crop, REF_NM_PER_PX, base_dose=2000.0,
                                         is_search=False, profile=ref_profile)

        search_x0 = search_y0 = None  # not meaningful -- different source
        search_px = min(int(round((SEARCH_PX * zoom_ratio) / REF_NM_PER_PX)), WORLD_PX)
        style_fn = find_containing_style_fn(regions, ref_x0, ref_y0, REF_PX) if absent_style_match else None
        search_source = build_absent_search_world(unique_seed, gen_mode, architecture,
                                                    forced_style_fn=style_fn, size=search_px,
                                                    is_color=(optical or hint))
        if search_source.shape[0] == search_px and search_source.shape[1] == search_px:
            # style_fn(search_px) path -- already exactly the right size
            search_world_crop = search_source
        else:
            # fallback-world path -- crop a random window out of it
            max_xy = max(0, search_source.shape[0] - search_px)
            crop_x0 = random.randint(0, max_xy)
            crop_y0 = random.randint(0, max_xy)
            search_world_crop = search_source[crop_y0:crop_y0 + search_px, crop_x0:crop_x0 + search_px]

        search_downsampled = cv2.resize(search_world_crop, (SEARCH_PX, SEARCH_PX), interpolation=cv2.INTER_AREA)
        search_profile = build_noise_profile(noise_severity, is_search=True,
                                              polygon_scale_pct_max=polygon_scale_pct_max)
        search_clean_render, _ = render_image(search_downsampled, zoom_ratio, base_dose=200.0,
                                               is_search=True, profile=search_profile)
        search_img, _ = rotate_search_image_and_point(search_clean_render, theta_deg, [(0.0, 0.0)])
        verify_info = {"ok": True, "err_px": float("nan"), "peak": float("nan"),
                       "second_peak": float("nan"), "margin": float("nan"), "attempts": 1}

    else:
        # Set A/B/D: retry pick_reference_crop() up to max_crop_attempts
        # times, keeping the first candidate whose reference crop is
        # verifiably unique in the rendered search image (verify_gt_unique,
        # v10 -- see module docstring). This guards against a periodic mat
        # crop that happens to correlate better somewhere else in the
        # search image than at its own true location, which would make
        # that pair's label unreproducible by any correct matcher.
        attempts = []
        chosen = None
        n_attempts = max(1, max_crop_attempts) if verify_crops else 1
        for _attempt_i in range(n_attempts):
            ref_x0, ref_y0, straddled, on_anchor = pick_reference_crop(regions, street_x_positions, mat_px)
            ref_world_crop = world[ref_y0:ref_y0 + REF_PX, ref_x0:ref_x0 + REF_PX]
            ref_profile = build_noise_profile(difficulty, is_search=False)
            reference_img, _ = render_image(ref_world_crop, REF_NM_PER_PX, base_dose=2000.0,
                                             is_search=False, profile=ref_profile)

            search_x0, search_y0, search_px = pick_search_window(ref_x0, ref_y0, zoom_ratio)
            search_world_crop = world[search_y0:search_y0 + search_px, search_x0:search_x0 + search_px]
            search_downsampled = cv2.resize(search_world_crop, (SEARCH_PX, SEARCH_PX), interpolation=cv2.INTER_AREA)
            search_profile = build_noise_profile(noise_severity, is_search=True,
                                                  polygon_scale_pct_max=polygon_scale_pct_max)
            search_clean_render, search_geom = render_image(search_downsampled, zoom_ratio, base_dose=200.0,
                                                              is_search=True, profile=search_profile)

            x_min = (ref_x0 - search_x0) / zoom_ratio
            y_min = (ref_y0 - search_y0) / zoom_ratio
            x_max = (ref_x0 + REF_PX - search_x0) / zoom_ratio
            y_max = (ref_y0 + REF_PX - search_y0) / zoom_ratio
            center_x_pre = (x_min + x_max) / 2.0
            center_y_pre = (y_min + y_max) / 2.0
            pts_pre = [(center_x_pre, center_y_pre),
                       (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

            # v10: push each pre-rotation point through the SAME geometric
            # distortions applied to the search pixels -- barrel distortion,
            # then raster drift, in the same order render_image() applies
            # them -- before the final post-render rotation forward-map.
            # Old code only forward-mapped through rotation, so GT_X/GT_Y
            # silently drifted by however far barrel_k/shear/jitter moved
            # the pixels underneath (worse at higher degradation severity).
            pts_post_barrel = [barrel_forward_pt(px, py, search_geom["barrel_k"], SEARCH_PX, SEARCH_PX)
                                for (px, py) in pts_pre]
            if search_geom["row_shift"] is not None:
                pts_post_drift = [drift_forward_pt(px, py, search_geom["row_shift"])
                                   for (px, py) in pts_post_barrel]
            else:
                pts_post_drift = pts_post_barrel

            search_img, mapped = rotate_search_image_and_point(search_clean_render, theta_deg, pts_post_drift)
            (center_x, center_y), corners = mapped[0], mapped[1:]

            gt_try = {"x": center_x, "y": center_y, "theta": theta_deg, "scale": zoom_ratio}
            if verify_crops:
                v = verify_gt_unique(reference_img, search_img, gt_try, tol_px=verify_tol_px)
            else:
                v = {"ok": True, "err_px": 0.0, "peak": float("nan"),
                     "second_peak": float("nan"), "margin": float("inf")}

            attempt_record = {
                "v": v, "reference_img": reference_img, "search_img": search_img,
                "gt": gt_try, "corners": corners, "ref_xy": (ref_x0, ref_y0),
                "search_xy": (search_x0, search_y0, search_px),
                "straddled": straddled, "on_anchor": on_anchor,
                "ref_profile": ref_profile, "search_profile": search_profile,
            }
            attempts.append(attempt_record)
            if v["ok"] and v["margin"] >= verify_good_margin:
                chosen = attempt_record
                break

        if chosen is None:
            # No attempt hit the "comfortable margin" bar. Prefer the best-
            # margin attempt that at least landed on its own label; only
            # fall back to the overall best-margin attempt (possibly not
            # "ok") if none did. This is deliberately non-fatal (a warning,
            # not a raise) -- a long batch run should never die because one
            # mat happened to be hard to place uniquely; the diagnostic is
            # kept in metadata (verify_ok/verify_margin) so it can be
            # filtered out downstream if desired.
            on_label = [a for a in attempts if a["v"]["ok"]]
            pool = on_label if on_label else attempts
            chosen = max(pool, key=lambda a: (a["v"]["margin"]
                                               if a["v"]["margin"] == a["v"]["margin"] else float("-inf")))
        reference_img = chosen["reference_img"]
        search_img = chosen["search_img"]
        center_x, center_y = chosen["gt"]["x"], chosen["gt"]["y"]
        corners = chosen["corners"]
        ref_x0, ref_y0 = chosen["ref_xy"]
        search_x0, search_y0, search_px = chosen["search_xy"]
        straddled, on_anchor = chosen["straddled"], chosen["on_anchor"]
        ref_profile, search_profile = chosen["ref_profile"], chosen["search_profile"]
        verify_info = dict(chosen["v"])
        verify_info["attempts"] = len(attempts)

    # NOTE (v9): Set D color is now native (see gen_mode selection above,
    # section 1e) -- no post-hoc gray->RGB tinting step here anymore.

    # Legacy Set-D rendering reused for local Set E.
    # The conversion is intentionally applied after the common SEM/noise/
    # rotation/ground-truth pipeline, matching generate_dataset.py.
    if hint:
        reference_img = render_optical_rgb(reference_img)
        search_img = render_optical_rgb(search_img)

    n_anchor_regions = sum(1 for r in regions if r["has_anchor"])
    n_twin_regions = sum(1 for r in regions if r.get("is_twin_duplicate"))
    styles_used = sorted(set(r["style"] for r in regions))
    level5, level5_name = difficulty_level_5(difficulty)

    citation_keys = sorted(set(
        k for s in styles_used for k in STYLE_CITATION_MAP.get(s, ([], ""))[0]
    ))

    metadata = {
        "world_id": world_id,
        "architecture_requested": architecture,
        "generation_mode": gen_mode,
        "difficulty": round(difficulty, 4),
        "difficulty_tier": ("easy" if difficulty < 1 / 3 else "medium" if difficulty < 2 / 3 else "hard"),
        "difficulty_level_5": level5,
        "difficulty_level_5_name": level5_name,
        "world_layout": "single_field" if gen_mode == "single_field" else "grid_streets",
        "straddles_boundary": straddled,
        "reference_centered_on_landmark": on_anchor,
        "mat_styles_in_world": styles_used,
        "n_landmark_regions_in_world": n_anchor_regions,
        "n_twin_duplicate_regions_in_world": n_twin_regions,
        "citation_keys": citation_keys,
        "search_noise_types": "+".join(search_profile["active_types"]),
        "search_dose_scale": round(search_profile["dose_scale"], 3),
        "search_shear_px": round(search_profile["shear_px"], 3),
        "search_jitter_px": round(search_profile["jitter_px"], 3),
        "search_beam_spot_nm": round(search_profile["beam_spot_nm"], 3),
        "search_edge_strength": round(search_profile["edge_strength"], 3),
        "search_polygon_scale_pct": search_profile["polygon_scale_pct"],
        "ref_noise_types": "+".join(ref_profile["active_types"]),
        "ref_edge_strength": round(ref_profile["edge_strength"], 3),
        "world_ref_crop_xy": [ref_x0, ref_y0],
        "world_search_window_xy": [search_x0, search_y0] if not is_absent else [None, None],
        # ---- Phase 2 pose / presence ground truth (register.py-contract-shaped) ----
        "true_instance_present": (not is_absent),
        "found": 0 if is_absent else 1,
        "is_degraded_pair": is_degraded,
        "degradation_severity_level": severity_level,
        "search_world_source": "different_world" if is_absent else "same_world",
        "optical_bonus": (optical or hint),
        "color_hint_bonus": False,
        # ---- v10: reference-crop uniqueness diagnostic (see verify_gt_unique) ----
        "verify_ok": bool(verify_info["ok"]),
        "verify_err_px": _finite_round(verify_info["err_px"], 3),
        "verify_peak": _finite_round(verify_info["peak"], 4),
        "verify_margin": _finite_round(verify_info["margin"], 4),
        "verify_attempts": verify_info.get("attempts", 1),
    }

    if is_absent:
        # Bug fix: these used to be hard-coded 0s, which is indistinguishable
        # from a real (0,0)-cornered box to any downstream code that doesn't
        # explicitly check found/true_instance_present first (e.g. a naive
        # PairDataset that crops GT_X_min:GT_X_max, GT_Y_min:GT_Y_max
        # unconditionally would silently manufacture an 8x8 "positive" patch
        # from the top-left corner of a search image that structurally
        # contains no true instance at all). NaN sentinels make that failure
        # loud (a crop with a NaN bound raises/produces empty, it doesn't
        # quietly succeed) instead of silent. Always branch on found /
        # true_instance_present before touching any GT_* pose field --
        # that is the correct signal, these are not a substitute for it.
        metadata.update({
            "GT_X": float("nan"), "GT_Y": float("nan"),
            "GT_theta_deg": float("nan"), "GT_scale": float("nan"),
            "GT_X_min": float("nan"), "GT_Y_min": float("nan"),
            "GT_X_max": float("nan"), "GT_Y_max": float("nan"),
            "GT_bbox_corners": None,
            "applied_search_zoom_ratio": round(zoom_ratio, 4),  # informational only, not graded
            "applied_search_theta_deg": round(theta_deg, 4),    # informational only, not graded
        })
    else:
        # Bug fix (v9): GT_X_min/Y_min/X_max/Y_max were previously written
        # from the PRE-rotation x_min/y_min/x_max/y_max -- stale as soon as
        # theta_deg != 0, since only GT_X/GT_Y (center) and GT_bbox_corners
        # were ever forward-mapped through the rotation matrix. Any code
        # that crops a "positive" patch out of search.png using the
        # GT_*_min/max columns was silently cropping the wrong region for
        # every rotated sample. Fixed here: GT_X_min/Y_min/X_max/Y_max are
        # now the axis-aligned bounding box of the ROTATED corners (post-
        # rotation, same frame as search.png on disk) -- a tight box that
        # fully contains the rotated quad. When theta_deg == 0 this is
        # numerically identical to the old (correct-by-luck) behavior, so
        # nothing changes for unrotated samples.
        corner_xs = [c[0] for c in corners]
        corner_ys = [c[1] for c in corners]
        bbox_x_min, bbox_x_max = min(corner_xs), max(corner_xs)
        bbox_y_min, bbox_y_max = min(corner_ys), max(corner_ys)
        metadata.update({
            "GT_X": round(center_x, 3), "GT_Y": round(center_y, 3),
            "GT_theta_deg": round(theta_deg, 4),
            "GT_scale": round(zoom_ratio, 4),
            "GT_X_min": round(bbox_x_min, 2), "GT_Y_min": round(bbox_y_min, 2),
            "GT_X_max": round(bbox_x_max, 2), "GT_Y_max": round(bbox_y_max, 2),
            "GT_bbox_corners": ";".join(f"{x:.2f},{y:.2f}" for x, y in corners),
        })
    return reference_img, search_img, metadata


def make_visualization(search_img, meta):
    vis = search_img.copy() if search_img.ndim == 3 else cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    if not meta.get("true_instance_present", True):
        h, w = vis.shape[:2]
        cv2.putText(vis, "ABSENT - no true instance", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    # Phase 2: GT region is a rotated quad, not an axis-aligned box, once
    # theta != 0 -- draw the actual rotated corners recorded in metadata.
    corners_str = meta.get("GT_bbox_corners")
    if corners_str:
        pts = np.array([[float(v) for v in pair.split(",")]
                         for pair in corners_str.split(";")], dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    else:
        p1 = (int(round(meta["GT_X_min"])), int(round(meta["GT_Y_min"])))
        p2 = (int(round(meta["GT_X_max"])), int(round(meta["GT_Y_max"])))
        cv2.rectangle(vis, p1, p2, (0, 0, 255), 2)

    cx, cy = int(round(meta["GT_X"])), int(round(meta["GT_Y"]))
    cv2.drawMarker(vis, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                    markerSize=16, thickness=2)
    return vis


# ============================================================================
# 7. CLI ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Drift-Sense synthetic SEM reference/search dataset generator. "
                     "Generates paired 1000x1000 reference (100x) and search (10x) "
                     "grayscale images with recorded ground-truth centre coordinates.")
    )
    parser.add_argument(
        "--architecture", type=str, default=DEFAULT_ARCHITECTURE,
        choices=["dram", "finfet", "mix"],
        help=("Which structure family to generate. 'dram' -> every pair is "
             "DRAM-style only. 'finfet' -> every pair is FinFET-style only. "
             "'mix' -> DRAM + FinFET + extra bonus pattern families (default: "
             f"{DEFAULT_ARCHITECTURE}). The hackathon problem statement asks "
             "you to pick DRAM-style OR FinFET-style for your submission.")
    )
    parser.add_argument(
        "--num_pairs", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of reference/search image pairs to generate (default: {DEFAULT_N_SAMPLES}). "
             "Problem statement requires at least 30 for validation."
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUT_ROOT,
        help=f"Directory to write the dataset into (default: {DEFAULT_OUT_ROOT})."
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help=f"Base random seed for reproducibility (default: {BASE_SEED})."
    )

    # ---- Phase 2 ("Registration under Unknown Pose") arguments ----
    parser.add_argument(
        "--set", type=str, default="mix", choices=list(SET_PRESETS.keys()),
        help=("Convenience preset matching the addendum's own Set A/B/C/D "
              "definitions, plus a local bonus Set E: A=nominal present, "
              "B=degraded present, C=absent (found=0), D=optical bonus "
              "(full native RGB), E=color-hint bonus (grayscale patterns "
              "with a sparse splash of colored accents). 'mix' (default) "
              "uses --absent-fraction/--degraded-fraction as given, "
              "approximating the organizer's ~70/70/40/20 blend.")
    )
    parser.add_argument(
        "--scale-min", type=float, default=DEFAULT_SCALE_MIN,
        help=f"Lower bound of the per-pair zoom ratio (default: {DEFAULT_SCALE_MIN})."
    )
    parser.add_argument(
        "--scale-max", type=float, default=DEFAULT_SCALE_MAX,
        help=f"Upper bound of the per-pair zoom ratio (default: {DEFAULT_SCALE_MAX})."
    )
    parser.add_argument(
        "--rotation-max-deg", type=float, default=DEFAULT_ROTATION_MAX_DEG,
        help=("Max +/- rotation in degrees applied to the search image "
              f"(default: {DEFAULT_ROTATION_MAX_DEG}). Set to 0 to disable rotation.")
    )
    parser.add_argument(
        "--absent-fraction", type=float, default=DEFAULT_ABSENT_FRACTION,
        help=("Fraction of pairs with NO true instance in the search image "
              f"(default: {DEFAULT_ABSENT_FRACTION}). Overridden by --set.")
    )
    parser.add_argument(
        "--degraded-fraction", type=float, default=DEFAULT_DEGRADED_FRACTION,
        help=("Of the PRESENT pairs, the fraction pushed through the severity "
              f"noise ladder rather than mild/nominal noise (default: {DEFAULT_DEGRADED_FRACTION}). "
              "Overridden by --set.")
    )
    parser.add_argument(
        "--severity-levels", type=int, default=DEFAULT_SEVERITY_LEVELS,
        help=f"Number of discrete degradation severity levels (default: {DEFAULT_SEVERITY_LEVELS})."
    )
    parser.add_argument(
        "--polygon-scale-pct-max", type=float, default=DEFAULT_POLYGON_SCALE_PCT_MAX,
        help=("Max +/- percent linewidth ('polygon') scaling applied to degraded "
              f"search images (default: {DEFAULT_POLYGON_SCALE_PCT_MAX}).")
    )
    parser.add_argument(
        "--optical", action="store_true",
        help="Force Set-D-style RGB optical-analogue rendering (also implied by --set D)."
    )
    parser.add_argument(
        "--hint", action="store_true",
        help=("Force the legacy Set-D pseudo-RGB rendering path used by Set E "
              "(also implied by --set E).")
    )
    parser.add_argument(
        "--no-absent-style-match", action="store_true",
        help=("Disable style-matched absent pairs (default: matched -- absent search "
              "images are rendered from the SAME pattern generator as the reference's own "
              "region, for a consistently 'plausible and periodically similar' negative set). "
              "Pass this to fall back to the old fully-random second-world behavior.")
    )
    parser.add_argument(
        "--set-wise", action="store_true",
        help=("Generate the FULL addendum shape in one run instead of a single "
              "set: Set_A/Set_B/Set_C/Set_D, each in its own subfolder under "
              "--output_dir, plus a blind pairs.csv (pair_id,reference_path,"
              "search_path -- what register.py actually receives) and a "
              "separate ground_truth.csv (full GT, for your own local scoring/"
              "threshold-tuning only -- never feed this to register.py). "
              "--num_pairs, --set, --absent-fraction, --degraded-fraction, "
              "--optical are ignored in this mode (the four sets' own presets "
              "are used). Counts default to the addendum's own 70/70/40/20 "
              "(Slide 3) -- override with --set-a-count etc. for a larger "
              "local training set; ratios don't need to match the organizer's "
              "blind set, since this data never leaves your machine.")
    )
    parser.add_argument("--set-a-count", type=int, default=70, help="Set A (nominal present) pair count.")
    parser.add_argument("--set-b-count", type=int, default=70, help="Set B (degraded present) pair count.")
    parser.add_argument("--set-c-count", type=int, default=40, help="Set C (absent) pair count.")
    parser.add_argument("--set-d-count", type=int, default=20, help="Set D (optical bonus) pair count.")

    # ---- Resumable / chunked set-wise generation (for time-limited runners) ----
    parser.add_argument(
        "--set-wise-build-part", type=str, default=None, choices=["A", "B", "C", "D"],
        help=("Build ONLY this one set (A/B/C/D) of the set-wise batch and stop -- "
              "does not write the final ground_truth.csv/pairs.csv. Intended for "
              "environments with a wall-clock limit per invocation: run this once "
              "per set (each picking up --set-a-count/etc. and --output_dir the "
              "same as a normal --set-wise call), then run --set-wise-assemble "
              "once all four parts are done. Writes an intermediate "
              "'_rows_<SET>.pkl' file under --output_dir that --set-wise-assemble "
              "reads back and cleans up. Uses the identical seed offsets as "
              "--set-wise (see build_set_wise_spec()), so the result is byte-for-"
              "byte the same dataset as a single --set-wise run with the same "
              "--set-a-count/etc. and --seed.")
    )
    parser.add_argument(
        "--set-wise-assemble", action="store_true",
        help=("Combine '_rows_A.pkl'.._rows_D.pkl' (previously written under "
              "--output_dir by --set-wise-build-part) into the final "
              "ground_truth.csv and pairs.csv, exactly like --set-wise would "
              "have produced in one run, then delete the intermediate pickles.")
    )

    # ---- v10: reference/ground-truth verification ----
    parser.add_argument(
        "--no-verify-crops", action="store_true",
        help=("Disable reference-crop uniqueness verification (default: enabled). "
              "Verification retries pick_reference_crop() against a template-match "
              "correlation search so a periodic mat can't silently produce an "
              "ambiguous/unreproducible label; disabling it is faster but drops "
              "that guarantee (restores old v9 single-shot behavior).")
    )
    parser.add_argument(
        "--max-crop-attempts", type=int, default=DEFAULT_MAX_CROP_ATTEMPTS,
        help=("Max reference-crop retries per present pair when verification is "
              f"enabled (default: {DEFAULT_MAX_CROP_ATTEMPTS}).")
    )
    parser.add_argument(
        "--verify-good-margin", type=float, default=DEFAULT_VERIFY_GOOD_MARGIN,
        help=("Correlation-peak margin over the best competing peak that stops "
              f"the retry loop early (default: {DEFAULT_VERIFY_GOOD_MARGIN}).")
    )
    parser.add_argument(
        "--verify-min-margin", type=float, default=DEFAULT_VERIFY_MIN_MARGIN,
        help=("Floor below which a fallback (non-early-stopped) crop choice gets "
              f"a console warning (default: {DEFAULT_VERIFY_MIN_MARGIN}). Non-fatal "
              "either way -- generation never raises over this.")
    )
    parser.add_argument(
        "--verify-tol-px", type=float, default=DEFAULT_VERIFY_TOL_PX,
        help=("Max pixel distance between the recorded GT centre and the "
              f"verification correlation peak to count as 'on label' (default: {DEFAULT_VERIFY_TOL_PX}).")
    )
    return parser.parse_args()


# ============================================================================
# 7b. PHASE 2 -- SET-WISE BATCH GENERATION (Set_A/B/C/D IN ONE RUN)
# ============================================================================

# (set_name, n_pairs, seed_offset) -- counts match the addendum's Slide 3
# breakdown: 70 nominal-present / 70 degraded-present / 40 absent /
# 20 optical-bonus = 200 blind pairs total. Distinct seed offsets keep
# Set_A/B/C/D from sharing underlying pattern "worlds".
def build_set_wise_spec(n_a=70, n_b=70, n_c=40, n_d=20):
    return [
        ("A", n_a, 0),
        ("B", n_b, 1_000_000),
        ("C", n_c, 2_000_000),
        ("D", n_d, 3_000_000),
    ]


def _build_one_set(set_name, n_pairs, seed_offset, base_seed, out_root, architecture,
                    scale_min, scale_max, rotation_max_deg, severity_levels,
                    polygon_scale_pct_max, verify_crops=DEFAULT_VERIFY_CROPS,
                    max_crop_attempts=DEFAULT_MAX_CROP_ATTEMPTS,
                    verify_good_margin=DEFAULT_VERIFY_GOOD_MARGIN,
                    verify_min_margin=DEFAULT_VERIFY_MIN_MARGIN,
                    verify_tol_px=DEFAULT_VERIFY_TOL_PX):
    absent_fraction, degraded_fraction, optical, hint = SET_PRESETS[set_name]
    set_dir = os.path.join(out_root, f"Set_{set_name}")
    os.makedirs(set_dir, exist_ok=True)

    rows = []
    print(f"\n== Set {set_name}: {n_pairs} pairs "
          f"(absent_fraction={absent_fraction}, degraded_fraction={degraded_fraction}, "
          f"optical={optical}, hint={hint}) ==")

    for i in range(n_pairs):
        pair_id = f"{set_name}_{i:04d}"
        sample_folder = os.path.join(set_dir, pair_id)
        os.makedirs(sample_folder, exist_ok=True)

        difficulty = i / max(1, n_pairs - 1)
        unique_seed = base_seed + seed_offset + i * 7919

        ref_img, search_img, meta = generate_sample(
            unique_seed, difficulty, architecture=architecture,
            scale_min=scale_min, scale_max=scale_max,
            rotation_max_deg=rotation_max_deg,
            absent_fraction=absent_fraction, degraded_fraction=degraded_fraction,
            severity_levels=severity_levels,
            polygon_scale_pct_max=polygon_scale_pct_max,
            optical=optical, hint=hint, absent_style_match=True,
            verify_crops=verify_crops, max_crop_attempts=max_crop_attempts,
            verify_good_margin=verify_good_margin, verify_min_margin=verify_min_margin,
            verify_tol_px=verify_tol_px)
        vis_img = make_visualization(search_img, meta)

        ref_rel = os.path.join(f"Set_{set_name}", pair_id, "reference.png")
        search_rel = os.path.join(f"Set_{set_name}", pair_id, "search.png")
        vis_rel = os.path.join(f"Set_{set_name}", pair_id, "visualization.png")

        cv2.imwrite(os.path.join(out_root, ref_rel), ref_img)
        cv2.imwrite(os.path.join(out_root, search_rel), search_img)
        cv2.imwrite(os.path.join(out_root, vis_rel), vis_img)

        row = {
            "pair_id": pair_id,
            "set": set_name,
            "reference_path": ref_rel,
            "search_path": search_rel,
            **{k: v for k, v in meta.items()
               if k not in ("mat_styles_in_world", "world_ref_crop_xy",
                            "world_search_window_xy", "GT_bbox_corners")},
        }
        rows.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == n_pairs:
            print(f"  {i + 1}/{n_pairs}")

    return rows


def run_set_wise_build_part(args):
    """Build exactly one set (A/B/C/D) of the --set-wise batch and stash its
    rows as a pickle under --output_dir, without writing the final CSVs.

    This lets the full --set-wise batch (default 200 pairs, ~8+ minutes) be
    split across several shorter invocations -- e.g. one shell/tool call per
    set -- in environments that impose a wall-clock limit per command. Once
    all four parts (A, B, C, D) have been built this way, call again with
    --set-wise-assemble to combine them into ground_truth.csv / pairs.csv
    identically to what a single --set-wise run would have produced (same
    seed offsets, same folder layout, same row schema).
    """
    set_name = args.set_wise_build_part
    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    spec = build_set_wise_spec(args.set_a_count, args.set_b_count, args.set_c_count, args.set_d_count)
    spec_by_name = {name: (n_pairs, seed_offset) for name, n_pairs, seed_offset in spec}
    n_pairs, seed_offset = spec_by_name[set_name]

    rows = _build_one_set(
        set_name, n_pairs, seed_offset, args.seed, out_root,
        args.architecture, args.scale_min, args.scale_max,
        args.rotation_max_deg, args.severity_levels,
        args.polygon_scale_pct_max,
        verify_crops=not args.no_verify_crops,
        max_crop_attempts=args.max_crop_attempts,
        verify_good_margin=args.verify_good_margin,
        verify_min_margin=args.verify_min_margin,
        verify_tol_px=args.verify_tol_px)

    pkl_path = os.path.join(out_root, f"_rows_{set_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(rows, f)

    print(f"\nSet {set_name}: {len(rows)} rows built and saved -> {pkl_path}")
    remaining = [n for n in ("A", "B", "C", "D")
                 if not os.path.exists(os.path.join(out_root, f"_rows_{n}.pkl"))]
    if remaining:
        print(f"Still need: {', '.join(remaining)}. "
              f"Run again with --set-wise-build-part <SET> for each, "
              f"then --set-wise-assemble once all four are done.")
    else:
        print("All four parts are present. Run with --set-wise-assemble to finish.")


def run_set_wise_assemble(args):
    """Combine '_rows_A.pkl'.._rows_D.pkl' (written by --set-wise-build-part)
    into the same ground_truth.csv / pairs.csv that a single --set-wise run
    would produce, then remove the intermediate pickles."""
    out_root = args.output_dir

    missing = [n for n in ("A", "B", "C", "D")
               if not os.path.exists(os.path.join(out_root, f"_rows_{n}.pkl"))]
    if missing:
        raise FileNotFoundError(
            f"Missing intermediate part(s) {missing} under '{out_root}'. "
            f"Run --set-wise-build-part for each of A/B/C/D first."
        )

    all_rows = []
    for set_name in ("A", "B", "C", "D"):
        pkl_path = os.path.join(out_root, f"_rows_{set_name}.pkl")
        with open(pkl_path, "rb") as f:
            all_rows.extend(pickle.load(f))

    df = pd.DataFrame(all_rows)

    gt_path = os.path.join(out_root, "ground_truth.csv")
    df.to_csv(gt_path, index=False)
    print(f"\nWrote local ground truth (do NOT feed to register.py) -> {gt_path}")

    blind_df = df[["pair_id", "reference_path", "search_path"]]
    pairs_path = os.path.join(out_root, "pairs.csv")
    blind_df.to_csv(pairs_path, index=False)
    print(f"Wrote blind contract input (what register.py actually sees) -> {pairs_path}")

    total = args.set_a_count + args.set_b_count + args.set_c_count + args.set_d_count
    print(f"\nTotal pairs: {len(df)} (target {total}: {args.set_a_count} A / "
          f"{args.set_b_count} B / {args.set_c_count} C / {args.set_d_count} D)")
    print(df["set"].value_counts())
    print("\nfound breakdown:")
    print(df["found"].value_counts())
    if "verify_ok" in df.columns:
        present = df[df["found"] == 1]
        n_not_ok = int((~present["verify_ok"]).sum())
        print(f"\nreference-crop verification: {n_not_ok}/{len(present)} present "
              f"pairs kept a below-floor crop after retries "
              f"(mean attempts={present['verify_attempts'].mean():.2f}, "
              f"mean verify_margin={present['verify_margin'].mean():.3f}).")

    for set_name in ("A", "B", "C", "D"):
        os.remove(os.path.join(out_root, f"_rows_{set_name}.pkl"))

    return df


def run_set_wise(args):
    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    spec = build_set_wise_spec(args.set_a_count, args.set_b_count, args.set_c_count, args.set_d_count)
    total = sum(n for _, n, _ in spec)
    print(f"Set-wise generation: A={args.set_a_count} B={args.set_b_count} "
          f"C={args.set_c_count} D={args.set_d_count} (total {total} pairs)")

    all_rows = []
    for set_name, n_pairs, seed_offset in spec:
        rows = _build_one_set(set_name, n_pairs, seed_offset, args.seed, out_root,
                               args.architecture, args.scale_min, args.scale_max,
                               args.rotation_max_deg, args.severity_levels,
                               args.polygon_scale_pct_max,
                               verify_crops=not args.no_verify_crops,
                               max_crop_attempts=args.max_crop_attempts,
                               verify_good_margin=args.verify_good_margin,
                               verify_min_margin=args.verify_min_margin,
                               verify_tol_px=args.verify_tol_px)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    gt_path = os.path.join(out_root, "ground_truth.csv")
    df.to_csv(gt_path, index=False)
    print(f"\nWrote local ground truth (do NOT feed to register.py) -> {gt_path}")

    blind_df = df[["pair_id", "reference_path", "search_path"]]
    pairs_path = os.path.join(out_root, "pairs.csv")
    blind_df.to_csv(pairs_path, index=False)
    print(f"Wrote blind contract input (what register.py actually sees) -> {pairs_path}")

    print(f"\nTotal pairs: {len(df)} (target {total}: {args.set_a_count} A / "
          f"{args.set_b_count} B / {args.set_c_count} C / {args.set_d_count} D)")
    print(df["set"].value_counts())
    print("\nfound breakdown:")
    print(df["found"].value_counts())
    if "verify_ok" in df.columns:
        present = df[df["found"] == 1]
        n_not_ok = int((~present["verify_ok"]).sum())
        print(f"\nv10 -- reference-crop verification: {n_not_ok}/{len(present)} present "
              f"pairs kept a below-floor crop after retries "
              f"(mean attempts={present['verify_attempts'].mean():.2f}, "
              f"mean verify_margin={present['verify_margin'].mean():.3f}).")
    return df


# ============================================================================
# 8. BATCH GENERATION WITH GROUPED FOLDERS
# ============================================================================

def main():
    args = parse_args()
    architecture = args.architecture
    n_samples = args.num_pairs
    out_root = args.output_dir
    base_seed = args.seed

    if args.set_wise_build_part is not None:
        return run_set_wise_build_part(args)

    if args.set_wise_assemble:
        return run_set_wise_assemble(args)

    if args.set_wise:
        return run_set_wise(args)

    # ---- Phase 2: resolve --set preset (if any) over the raw flags ----
    absent_fraction = args.absent_fraction
    degraded_fraction = args.degraded_fraction
    optical = args.optical
    hint = args.hint
    preset = SET_PRESETS.get(args.set)
    if preset is not None:
        absent_fraction, degraded_fraction, preset_optical, preset_hint = preset
        optical = optical or preset_optical
        hint = hint or preset_hint

    if n_samples < 2:
        raise ValueError("--num_pairs must be at least 2 (difficulty ramps from 0.0 to 1.0 across samples).")

    os.makedirs(out_root, exist_ok=True)

    records = []
    search_hashes = set()
    duplicate_count = 0

    print(f"Generating {n_samples} '{architecture}' samples (set={args.set}) in grouped folders under '{out_root}'...")
    print(f"  scale range: [{args.scale_min}, {args.scale_max}]x | rotation: +/-{args.rotation_max_deg} deg | "
          f"absent_fraction={absent_fraction} | degraded_fraction={degraded_fraction} | "
          f"severity_levels={args.severity_levels} | optical={optical} | hint={hint}")
    print(f"  verify_crops={not args.no_verify_crops} | max_crop_attempts={args.max_crop_attempts} | "
          f"verify_good_margin={args.verify_good_margin} | verify_min_margin={args.verify_min_margin}")

    for i in range(n_samples):
        sample_id = f"sample_{i:04d}"
        sample_folder = os.path.join(out_root, sample_id)
        os.makedirs(sample_folder, exist_ok=True)

        difficulty = i / (n_samples - 1)
        unique_seed = base_seed + i * 7919

        ref_img, search_img, meta = generate_sample(
            unique_seed, difficulty, architecture=architecture,
            scale_min=args.scale_min, scale_max=args.scale_max,
            rotation_max_deg=args.rotation_max_deg,
            absent_fraction=absent_fraction, degraded_fraction=degraded_fraction,
            severity_levels=args.severity_levels,
            polygon_scale_pct_max=args.polygon_scale_pct_max,
            optical=optical, hint=hint, absent_style_match=not args.no_absent_style_match,
            verify_crops=not args.no_verify_crops, max_crop_attempts=args.max_crop_attempts,
            verify_good_margin=args.verify_good_margin, verify_min_margin=args.verify_min_margin,
            verify_tol_px=args.verify_tol_px)
        vis_img = make_visualization(search_img, meta)

        search_hash = hash(search_img.tobytes())
        if search_hash in search_hashes:
            duplicate_count += 1
        search_hashes.add(search_hash)

        ref_rel_path = os.path.join(sample_id, "reference.png")
        search_rel_path = os.path.join(sample_id, "search.png")
        vis_rel_path = os.path.join(sample_id, "visualization.png")
        meta_rel_path = os.path.join(sample_id, "metadata.json")

        cv2.imwrite(os.path.join(out_root, ref_rel_path), ref_img)
        cv2.imwrite(os.path.join(out_root, search_rel_path), search_img)
        cv2.imwrite(os.path.join(out_root, vis_rel_path), vis_img)

        sample_meta = {"sample_id": sample_id, **meta}
        with open(os.path.join(out_root, meta_rel_path), "w") as f:
            json.dump(sample_meta, f, indent=4)

        row = {
            "sample_id": sample_id,
            "reference_file": ref_rel_path,
            "search_file": search_rel_path,
            "visualization_file": vis_rel_path,
            **{k: (v if not isinstance(v, list) else "+".join(v))
               for k, v in meta.items() if k not in ("mat_styles_in_world",
                                                       "world_ref_crop_xy",
                                                       "world_search_window_xy")},
        }
        records.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  Progress: {i + 1}/{n_samples} samples created.")

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_root, "ground_truth.csv")
    df.to_csv(csv_path, index=False)

    citations_manifest = {
        "architecture_requested": architecture,
        "citations": CITATIONS,
        "style_citation_map": {
            style: {"citation_keys": keys, "justification": note}
            for style, (keys, note) in STYLE_CITATION_MAP.items()
        },
        "difficulty_levels": {
            name: {"level": level, "range": [lo, hi]}
            for lo, hi, level, name in DIFFICULTY_LEVEL_5_BOUNDS
        },
        "note": ("These sources provide technology and layout context that motivates "
                 "the synthetic image generation strategy (pattern style, density, "
                 "periodicity). They are not claimed as sources of literal numeric "
                 "image-generation parameters (pitch, noise sigma, etc.). Edge-"
                 "brightening and the DRAM word-line/bit-line-with-via-at-intersection "
                 "and FinFET fin-lines-with-gate-bars structures follow the sponsor's "
                 "own sample-prompt description (Applied Materials Drift-Sense problem "
                 "statement), not a separate literature source."),
    }
    with open(os.path.join(out_root, "citations.json"), "w") as f:
        json.dump(citations_manifest, f, indent=2)
    print(f"\nWrote citations manifest -> {os.path.join(out_root, 'citations.json')}")

    print(f"\nGenerated {len(df)} samples (architecture='{architecture}').")
    print(f"Unique search images: {len(search_hashes)} / {len(df)} "
          f"({'OK, all unique' if duplicate_count == 0 else f'{duplicate_count} duplicates found!'})")
    print(f"Unique underlying worlds: {df['world_id'].nunique()} / {len(df)} samples")
    print("\nGeneration-mode breakdown:")
    print(df["generation_mode"].value_counts())
    print("\nreference_centered_on_landmark breakdown:")
    print(df["reference_centered_on_landmark"].value_counts())
    print("\ndifficulty_level_5_name breakdown:")
    print(df["difficulty_level_5_name"].value_counts())
    print("\nPhase 2 -- found (1=present, 0=absent) breakdown:")
    print(df["found"].value_counts())
    print("\nPhase 2 -- is_degraded_pair breakdown (of present pairs):")
    print(df.loc[df["found"] == 1, "is_degraded_pair"].value_counts())
    print("\nPhase 2 -- degradation_severity_level breakdown (0 = not degraded):")
    print(df["degradation_severity_level"].value_counts().sort_index())
    if "verify_ok" in df.columns:
        present = df[df["found"] == 1]
        n_not_ok = int((~present["verify_ok"]).sum())
        print(f"\nv10 -- reference-crop verification: {n_not_ok}/{len(present)} present "
              f"pairs kept a below-floor crop after retries "
              f"(mean attempts={present['verify_attempts'].mean():.2f}, "
              f"mean verify_margin={present['verify_margin'].mean():.3f}).")

    zip_path = shutil.make_archive(out_root, "zip", out_root)
    print(f"\nZipped dataset -> {zip_path}")

    if IN_COLAB:
        print("Triggering browser download...")
        files.download(f"{out_root}.zip")
    else:
        print(f"Dataset saved locally at: {os.path.abspath(zip_path)}")

    return df


if __name__ == "__main__":
    main()