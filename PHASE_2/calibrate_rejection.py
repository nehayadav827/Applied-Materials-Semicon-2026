"""
training/calibrate_rejection.py -- calibrates the found/score threshold
that register.py uses (FOUND_THRESHOLD constant). Run this against your
own present (Set A/B-style) vs absent (Set C-style) data, then update
register.py's FOUND_THRESHOLD (or pass --found-threshold at call time)
with the printed best value.

Usage:
    python training/calibrate_rejection.py --present out_setA out_setB --absent out_setC
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matching import generate_candidates


def top_scores(root, tag=""):
    df = pd.read_csv(os.path.join(root, "ground_truth.csv"))
    scores = []
    t0 = time.time()
    for i, row in df.iterrows():
        ref = cv2.imread(os.path.join(root, row["reference_file"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(root, row["search_file"]), cv2.IMREAD_GRAYSCALE)
        cands = generate_candidates(ref, search, top_k=1)
        scores.append(cands[0][0] if cands else -1.0)
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(df) - (i + 1))
            print(f"[{tag}] {i+1}/{len(df)}  elapsed={elapsed:.0f}s  ~{rate:.2f}s/img  eta={eta:.0f}s")
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--present", nargs="+", required=True)
    parser.add_argument("--absent", nargs="+", required=True)
    args = parser.parse_args()

    present_scores = np.concatenate([top_scores(r, tag=f"present:{r}") for r in args.present])
    absent_scores = np.concatenate([top_scores(r, tag=f"absent:{r}") for r in args.absent])
    print(f"present: n={len(present_scores)} mean={present_scores.mean():.3f} min={present_scores.min():.3f}")
    print(f"absent:  n={len(absent_scores)} mean={absent_scores.mean():.3f} max={absent_scores.max():.3f}")

    best_thresh, best_f1 = None, -1.0
    for thresh in np.linspace(0.0, 1.0, 201):
        tp = (present_scores >= thresh).sum()
        fn = (present_scores < thresh).sum()
        fp = (absent_scores >= thresh).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"\nBest threshold={best_thresh:.3f}  F1={best_f1:.3f}")
    print("Update register.py's FOUND_THRESHOLD (or pass --found-threshold) with this value.")


if __name__ == "__main__":
    main()
