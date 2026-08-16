"""
================================================================================
 Multi-Region Die-Layout SEM Reference/Search BATCH Generator (v7 - Drift-Sense)
================================================================================
Standalone dataset-generator script for the Applied Materials "Drift-Sense"
hackathon problem statement. Generates paired 1000x1000 grayscale reference
(100x close-up) / search (10x wide-field) SEM-style images with recorded
ground-truth target-centre coordinates and per-pair metadata.

Usage:
    python generate_dataset.py --architecture dram   --num_pairs 50 --output_dir ./out_dram
    python generate_dataset.py --architecture finfet --num_pairs 50 --output_dir ./out_finfet
    python generate_dataset.py --architecture mix     --num_pairs 50 --output_dir ./out_mix
    python generate_dataset.py --help

Required CLI parameters (per the hackathon deliverables table):
  --architecture   dram | finfet | mix   (which structure family to generate)
  --num_pairs      int                   (number of pairs to generate, >=30 recommended)
  --output_dir     path                  (directory the dataset is written into)

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

MAT_SIZE_NM = 2600
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
DEFAULT_N_SAMPLES = 50
DEFAULT_OUT_ROOT = "./synthetic_sem_dataset_holdout"
DEFAULT_ARCHITECTURE = "mix"


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

def build_world_grid(family, difficulty=0.5, architecture="mix"):
    style_fns = get_family_style_fns(family, architecture)
    if family == "legacy":
        anchor_prob = ANCHOR_PROB_LEGACY
    elif family == "realistic":
        anchor_prob = ANCHOR_PROB_REALISTIC
    else:
        anchor_prob = ANCHOR_PROB_EXPANDED

    canvas = np.full((WORLD_PX, WORLD_PX), STREET_GRAY, dtype=np.uint8)
    mat_px = int(MAT_SIZE_NM / REF_NM_PER_PX)
    street_px = int(STREET_WIDTH_NM / REF_NM_PER_PX)
    positions = [street_px + i * (mat_px + street_px) for i in range(GRID_N)]

    duplicate_prob = lerp(0.0, 0.75, max(0.0, (difficulty - 0.4) / 0.6))
    twin_pattern, twin_params, twin_mat_indices = None, None, set()
    n_mats = GRID_N * GRID_N
    if random.random() < duplicate_prob:
        twin_pattern, twin_params = random.choice(style_fns)(mat_px)
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
            else:
                pattern, params = random.choice(style_fns)(mat_px)
                params["is_twin_duplicate"] = False

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


def build_world_single_field(architecture="mix"):
    style_fns = get_single_field_style_fns(architecture)
    canvas, params = random.choice(style_fns)(WORLD_PX)

    has_anchor = random.random() < ANCHOR_PROB_SINGLE_FIELD
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


def apply_raster_drift(img, shear_px, jitter_px):
    if shear_px == 0 and jitter_px == 0:
        return img
    h, w = img.shape[:2]
    rows = np.arange(h, dtype=np.float32)
    shear_shift = shear_px * (rows / h)
    jitter = np.random.normal(0, jitter_px, size=h).astype(np.float32)
    row_shift = shear_shift + jitter
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


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


def build_noise_profile(difficulty, is_search):
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
        "active_types": active_types,
    }
    return profile


def render_image(world_crop, nm_per_px, base_dose, is_search, profile):
    img = world_crop.copy()
    img = apply_geometric_distortion(img, profile["barrel_k"])

    round_sigma = max(0.3, 1.0 / nm_per_px)
    img = apply_cd_bias_and_rounding(img, 0.0, round_sigma)

    sigma_px = profile["beam_spot_nm"] / nm_per_px
    img = apply_beam_blur(img, sigma_px, profile["astig_ratio"])

    if is_search:
        img = apply_raster_drift(img, shear_px=profile["shear_px"], jitter_px=profile["jitter_px"])

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

    return img


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


def pick_search_window(ref_x0, ref_y0):
    search_px = int((SEARCH_PX * SEARCH_NM_PER_PX) / REF_NM_PER_PX)
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


def generate_sample(unique_seed, difficulty, architecture="mix"):
    random.seed(unique_seed)
    np.random.seed(unique_seed % (2**32 - 1))

    gen_mode = choose_generation_mode(architecture)
    if gen_mode == "single_field":
        world, regions, street_x_positions, mat_px, street_px = build_world_single_field(architecture)
    else:
        world, regions, street_x_positions, mat_px, street_px = build_world_grid(
            gen_mode, difficulty=difficulty, architecture=architecture)

    world_id = compute_world_id(gen_mode, unique_seed, regions)

    ref_x0, ref_y0, straddled, on_anchor = pick_reference_crop(regions, street_x_positions, mat_px)
    search_x0, search_y0, search_px = pick_search_window(ref_x0, ref_y0)

    ref_world_crop = world[ref_y0:ref_y0 + REF_PX, ref_x0:ref_x0 + REF_PX]
    ref_profile = build_noise_profile(difficulty, is_search=False)
    reference_img = render_image(ref_world_crop, REF_NM_PER_PX, base_dose=2000.0,
                                  is_search=False, profile=ref_profile)

    search_world_crop = world[search_y0:search_y0 + search_px, search_x0:search_x0 + search_px]
    search_downsampled = cv2.resize(search_world_crop, (SEARCH_PX, SEARCH_PX), interpolation=cv2.INTER_AREA)
    search_profile = build_noise_profile(difficulty, is_search=True)
    search_img = render_image(search_downsampled, SEARCH_NM_PER_PX, base_dose=200.0,
                               is_search=True, profile=search_profile)

    x_min = (ref_x0 - search_x0) / SCALE_RATIO
    y_min = (ref_y0 - search_y0) / SCALE_RATIO
    x_max = (ref_x0 + REF_PX - search_x0) / SCALE_RATIO
    y_max = (ref_y0 + REF_PX - search_y0) / SCALE_RATIO

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

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
        "ref_noise_types": "+".join(ref_profile["active_types"]),
        "ref_edge_strength": round(ref_profile["edge_strength"], 3),
        "world_ref_crop_xy": [ref_x0, ref_y0],
        "world_search_window_xy": [search_x0, search_y0],
        "GT_X_min": round(x_min, 2), "GT_Y_min": round(y_min, 2),
        "GT_X_max": round(x_max, 2), "GT_Y_max": round(y_max, 2),
        "GT_X": round(center_x, 2), "GT_Y": round(center_y, 2),
    }
    return reference_img, search_img, metadata


def make_visualization(search_img, meta):
    vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
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
    return parser.parse_args()


# ============================================================================
# 8. BATCH GENERATION WITH GROUPED FOLDERS
# ============================================================================

def main():
    args = parse_args()
    architecture = args.architecture
    n_samples = args.num_pairs
    out_root = args.output_dir
    base_seed = args.seed

    if n_samples < 2:
        raise ValueError("--num_pairs must be at least 2 (difficulty ramps from 0.0 to 1.0 across samples).")

    os.makedirs(out_root, exist_ok=True)

    records = []
    search_hashes = set()
    duplicate_count = 0

    print(f"Generating {n_samples} '{architecture}' samples in grouped folders under '{out_root}'...")

    for i in range(n_samples):
        sample_id = f"sample_{i:04d}"
        sample_folder = os.path.join(out_root, sample_id)
        os.makedirs(sample_folder, exist_ok=True)

        difficulty = i / (n_samples - 1)
        unique_seed = base_seed + i * 7919

        ref_img, search_img, meta = generate_sample(unique_seed, difficulty, architecture=architecture)
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
