"""
Drift-Sense localization inference script -- THE script Applied Materials
runs directly on their test image pairs.

Usage:
    python localize.py --reference path/to/reference.png --search path/to/search.png

Output contract (important for automated grading):
    The LAST line printed to stdout is exactly "x,y" -- the predicted
    center of the reference pattern within the search image, in
    search-image pixel coordinates (origin top-left, x right, y down),
    and NOTHING else is on that line. All other messages (timing,
    diagnostics) go to stderr, so a harness that reads stdout and parses
    the last line, or the only line if run with --quiet, always gets a
    clean coordinate.

    Pass --json to instead print a single JSON object to stdout:
        {"x": 512.34, "y": 488.10, "runtime_s": 0.842}

No manual edits are required. matching.py must be present in the same
directory (it is -- this repo ships both together). If reranker.pt is
also present in the same directory (or a path is given via --reranker),
you can opt into the CNN-hybrid mode with --use-cnn -- but this is OFF
by default because, on our own validation, classical CV + sub-pixel
refinement alone consistently outperformed the CNN reranker (78.6% vs
~25-30% within 5px on the disambiguating-context population; see
README.md and the training/evaluation scripts for the full comparison).
torch is only imported if --use-cnn is passed, so this script has zero
torch dependency in its default, best-performing mode.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from matching import generate_candidates


def log(*a, **kw):
    """All non-coordinate output goes to stderr, keeping stdout clean for
    whatever harness parses the final coordinate line."""
    print(*a, file=sys.stderr, **kw)


def load_gray(path, label):
    if not os.path.isfile(path):
        log(f"ERROR: {label} image not found: {path}")
        sys.exit(2)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"ERROR: could not read {label} image (unsupported/corrupt file?): {path}")
        sys.exit(2)
    return img


def localize_cv_only(reference, search):
    """Verified-best mode: plain CV top-1 pick (candidates are already
    best-first by score and sub-pixel refined inside generate_candidates).
    context-bucket result on our 500-pair dataset's held-out test split:
    within_5px=78.6%, within_2px=67.9%, median_err=0.83px."""
    candidates = generate_candidates(reference, search, top_k=30)
    if not candidates:
        h, w = search.shape[:2]
        log("WARNING: no candidates found at all -- falling back to search-image center.")
        return w / 2.0, h / 2.0
    return candidates[0][1], candidates[0][2]


def localize_hybrid_cnn(reference, search, reranker_path):
    """Optional CNN-reranked mode, only reached with --use-cnn. Imports
    torch lazily so the default CV-only path has no torch dependency."""
    import torch
    from train_reranker import TinyEmbedNet, PATCH_SIZE

    def to_patch_tensor(img_patch):
        p = cv2.resize(img_patch, (PATCH_SIZE, PATCH_SIZE))
        return torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0) / 255.0

    def to_ref_tensor(reference, scale):
        h, w = reference.shape[:2]
        pre_w = max(8, int(round(w / scale)))
        pre_h = max(8, int(round(h / scale)))
        p = cv2.resize(reference, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
        p = cv2.resize(p, (PATCH_SIZE, PATCH_SIZE))
        return torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0) / 255.0

    if not os.path.isfile(reranker_path):
        log(f"ERROR: --use-cnn was given but reranker weights not found: {reranker_path}")
        sys.exit(2)

    model = TinyEmbedNet(embed_dim=128)
    model.load_state_dict(torch.load(reranker_path, map_location="cpu"))
    model.eval()

    h, w = search.shape[:2]
    search_center = (w / 2.0, h / 2.0)
    candidates = generate_candidates(reference, search, top_k=30)
    if not candidates:
        log("WARNING: no candidates found at all -- falling back to search-image center.")
        return search_center

    with torch.no_grad():
        ref_emb_cache = {}

        def ref_embedding(scale):
            if scale not in ref_emb_cache:
                ref_emb_cache[scale] = model(to_ref_tensor(reference, scale))
            return ref_emb_cache[scale]

        scored = []
        for score, cx, cy, scale, angle in candidates:
            pw = max(int(reference.shape[1] / scale), 8)
            ph = max(int(reference.shape[0] / scale), 8)
            x0 = max(0, int(cx - pw / 2))
            y0 = max(0, int(cy - ph / 2))
            patch = search[y0:y0 + ph, x0:x0 + pw]
            if patch.size == 0:
                continue
            emb = model(to_patch_tensor(patch))
            dist = (ref_embedding(scale) - emb).pow(2).sum().item()
            scored.append((dist, cx, cy))

    if not scored:
        return search_center

    scored.sort(key=lambda s: s[0])
    best_dist = scored[0][0]
    CENTER_TIE_MARGIN = 0.02
    valid = [s for s in scored if s[0] <= best_dist + CENTER_TIE_MARGIN]
    valid.sort(key=lambda s: (s[1] - search_center[0]) ** 2 + (s[2] - search_center[1]) ** 2)
    _, cx, cy = valid[0]
    return cx, cy


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Path to the reference (100x) image.")
    parser.add_argument("--search", required=True, help="Path to the search (10x) image.")
    parser.add_argument("--use-cnn", action="store_true",
                         help="Opt into the CNN-reranked hybrid mode. OFF by default -- "
                              "see docstring for why.")
    parser.add_argument("--reranker", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reranker.pt"),
                         help="Path to reranker weights, only used with --use-cnn.")
    parser.add_argument("--json", action="store_true",
                         help="Print a JSON object instead of a bare 'x,y' line.")
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress all stderr diagnostic output too.")
    args = parser.parse_args()

    if args.quiet:
        global log
        log = lambda *a, **kw: None

    reference = load_gray(args.reference, "reference")
    search = load_gray(args.search, "search")

    t0 = time.perf_counter()
    if args.use_cnn:
        log("Mode: CV + CNN hybrid reranker")
        cx, cy = localize_hybrid_cnn(reference, search, args.reranker)
    else:
        log("Mode: classical CV + sub-pixel refinement (verified-best, no CNN)")
        cx, cy = localize_cv_only(reference, search)
    elapsed = time.perf_counter() - t0

    log(f"Runtime: {elapsed:.3f}s")

    if args.json:
        print(json.dumps({"x": round(float(cx), 3), "y": round(float(cy), 3),
                           "runtime_s": round(elapsed, 4)}))
    else:
        print(f"{cx:.3f},{cy:.3f}")


if __name__ == "__main__":
    main()
