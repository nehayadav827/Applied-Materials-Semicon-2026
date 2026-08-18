import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
from matching import generate_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./synthetic_sem_dataset")
    parser.add_argument("--out", default="hard_negatives.csv")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dataset, "ground_truth.csv"))
    rows = []
    t_start = time.time()

    for i, row in df.iterrows():
        ref = cv2.imread(os.path.join(args.dataset, row["reference_file"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(args.dataset, row["search_file"]), cv2.IMREAD_GRAYSCALE)
        gt_x, gt_y = row["GT_X"], row["GT_Y"]
        pos_w = max(int(row["GT_X_max"] - row["GT_X_min"]), 8)
        pos_h = max(int(row["GT_Y_max"] - row["GT_Y_min"]), 8)

        candidates = generate_candidates(ref, search, top_k=30)

        neg_x, neg_y = gt_x + pos_w * 2, gt_y   # fallback
        for score, cx, cy, scale, angle in candidates:
            if abs(cx - gt_x) > pos_w or abs(cy - gt_y) > pos_h:
                neg_x, neg_y = cx, cy
                break

        rows.append({"sample_id": row["sample_id"], "neg_x": neg_x, "neg_y": neg_y})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"[{i+1}/{len(df)}] elapsed={elapsed:.0f}s est_total={elapsed/(i+1)*len(df):.0f}s")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} hard negatives to {args.out}")

if __name__ == "__main__":
    main()