"""
Final evaluation script -- produces the manifest, pixel pass rates,
runtime/hardware report, and failure case required for submission.

By default this scores ONLY the held-out test split written by
train_reranker.py (dataset/test_split.csv) -- samples from worlds the
reranker never saw during training or model selection. That's the number
that belongs in the PPT/README. Use --split all to report on the whole
dataset instead (useful for a "how good is the CV baseline in general"
sanity table, but do not present that as the held-out accuracy).

Coordinate convention: origin (0,0) is the TOP-LEFT of the search image,
x increases right, y increases down (standard OpenCV/numpy array convention).
GT_X, GT_Y in ground_truth.csv are the reference bounding box CENTER in
search-image pixel coordinates, computed at 10:1 scale by the generator.

Usage:
    python evaluate.py --dataset ./synthetic_sem_dataset --reranker reranker.pt
    python evaluate.py --dataset ./synthetic_sem_dataset --split all   # whole-dataset sanity check
"""

import argparse
import os
import platform
import time
import cv2
import numpy as np
import pandas as pd
import torch
from matching import generate_candidates
from train_reranker import TinyEmbedNet, PATCH_SIZE

CENTER_TIE_MARGIN = 0.02   # candidates within 2% embedding-distance of the
                            # best are treated as "valid matches" for the
                            # closest-to-centre rule (spec section: when
                            # multiple valid matches exist, pick the one
                            # closest to the search-image center)


def to_patch_tensor(img_patch):
    p = cv2.resize(img_patch, (PATCH_SIZE, PATCH_SIZE))
    return torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0) / 255.0


def to_ref_tensor(reference, scale):
    """Match train_reranker.py's fix: pre-downsample the full-resolution
    reference by the candidate's own scale before the final PATCH_SIZE
    resize, so the reference and the candidate patch go through a
    comparable amount of detail loss instead of the reference being
    crushed ~10x harder than the candidate."""
    h, w = reference.shape[:2]
    pre_w = max(8, int(round(w / scale)))
    pre_h = max(8, int(round(h / scale)))
    p = cv2.resize(reference, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
    p = cv2.resize(p, (PATCH_SIZE, PATCH_SIZE))
    return torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0) / 255.0


def localize_one(reference, search, model):
    h, w = search.shape[:2]
    search_center = (w / 2.0, h / 2.0)

    candidates = generate_candidates(reference, search, top_k=30)
    if not candidates:
        return w / 2.0, h / 2.0  # degenerate fallback, should not normally hit

    if model is None:
        # --no-cnn ablation: trust the CV ranking as-is (already best-first,
        # sub-pixel refined). Use this to sanity-check whether the reranker
        # is actually helping before trusting it in the final pipeline.
        return candidates[0][1], candidates[0][2]

    with torch.no_grad():
        ref_emb_cache = {}   # scale -> embedding, since each candidate can carry a
                              # slightly different scale and the reference must be
                              # pre-downsampled to match that specific candidate's scale

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

    # closest-to-centre rule: among candidates within CENTER_TIE_MARGIN of
    # the best embedding distance, pick the one nearest the search-image center
    valid = [s for s in scored if s[0] <= best_dist + CENTER_TIE_MARGIN]
    valid.sort(key=lambda s: (s[1] - search_center[0]) ** 2 + (s[2] - search_center[1]) ** 2)

    _, cx, cy = valid[0]
    return cx, cy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./synthetic_sem_dataset")
    parser.add_argument("--reranker", default="reranker.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["test", "all"],
                         help="'test' (default) scores only the held-out split written by "
                              "train_reranker.py; 'all' scores the whole dataset (sanity/ablation only)")
    parser.add_argument("--no-cnn", action="store_true",
                         help="skip the reranker, score the raw CV top-1 pick. Run this "
                              "and compare to the default hybrid mode -- if --no-cnn scores "
                              "BETTER, the reranker is currently hurting, not helping.")
    parser.add_argument("--out_csv", default="evaluation_manifest.csv")
    parser.add_argument("--quiet", action="store_true",
                         help="Suppress the per-sample pixel-error line (still printed by "
                              "default). Useful for very large test splits where you only "
                              "want the final summary.")
    args = parser.parse_args()

    model = None
    if not args.no_cnn:
        model = TinyEmbedNet(embed_dim=128)
        model.load_state_dict(torch.load(args.reranker, map_location="cpu"))
        model.eval()

    df = pd.read_csv(os.path.join(args.dataset, "ground_truth.csv"))

    if args.split == "test":
        split_path = os.path.join(args.dataset, "test_split.csv")
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"{split_path} not found. Run train_reranker.py first -- it writes the "
                f"held-out test sample_ids there. Or pass --split all to score the whole "
                f"dataset instead (not a valid held-out number, sanity/ablation use only).")
        test_ids = pd.read_csv(split_path)["sample_id"]
        df = df[df["sample_id"].isin(test_ids)].reset_index(drop=True)
        print(f"Scoring held-out TEST split only: {len(df)} samples "
              f"(worlds never seen during training or model selection)")
    else:
        print(f"Scoring WHOLE dataset ({len(df)} samples) -- includes train/val worlds, "
              f"NOT a valid generalization estimate, sanity/ablation use only")

    if args.limit:
        df = df.head(args.limit)

    print("Hardware:", platform.processor(), "| device=cpu",
          "| python=" + platform.python_version(),
          "| torch=" + torch.__version__, "| opencv=" + cv2.__version__)
    print("Timing method: wall-clock time.perf_counter() around candidate "
          "generation + CNN rerank + centre tie-break, per sample.\n")

    rows = []
    times = []

    for i, row in df.iterrows():
        ref = cv2.imread(os.path.join(args.dataset, row["reference_file"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(args.dataset, row["search_file"]), cv2.IMREAD_GRAYSCALE)

        t0 = time.perf_counter()
        pred_x, pred_y = localize_one(ref, search, model)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        gt_x, gt_y = row["GT_X"], row["GT_Y"]
        err = float(np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2))

        rows.append({
            "sample_id": row["sample_id"],
            "reference_file": row["reference_file"],
            "search_file": row["search_file"],
            "difficulty_level_5_name": row.get("difficulty_level_5_name", ""),
            "generation_mode": row.get("generation_mode", ""),
            "gt_x": gt_x, "gt_y": gt_y,
            "pred_x": round(pred_x, 2), "pred_y": round(pred_y, 2),
            "pixel_error": round(err, 2),
            "pass_5px": err <= 5, "pass_4px": err <= 4,
            "pass_2px": err <= 2, "pass_1px": err <= 1,
            "runtime_s": round(elapsed, 4),
        })

        if not args.quiet:
            print(f"[{i+1:4d}/{len(df)}] {row['sample_id']:16s} "
                  f"pred=({pred_x:7.2f},{pred_y:7.2f})  "
                  f"gt=({gt_x:7.2f},{gt_y:7.2f})  "
                  f"error={err:8.2f}px  {elapsed:.3f}s")


        if (i + 1) % 50 == 0:
            print(f"  -- running mean_err={np.mean([r['pixel_error'] for r in rows]):.2f}px "
                  f"over {i+1} samples --")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    errors = out_df["pixel_error"].values
    print("\n--- FINAL RESULTS ---")
    print(f"Split: {args.split}")
    print(f"Samples evaluated: {len(out_df)}")
    print(f"Mean error:   {errors.mean():.2f} px")
    print(f"Median error: {np.median(errors):.2f} px")
    for thresh in [5, 4, 2, 1]:
        pct = (errors <= thresh).mean() * 100
        print(f"  within {thresh}px: {pct:.1f}%")
    print(f"Avg runtime: {np.mean(times):.3f}s/sample  ({sum(times):.1f}s total compute)")

    if "difficulty_level_5_name" in out_df.columns:
        print("\n--- Accuracy by difficulty tier ---")
        for tier, group in out_df.groupby("difficulty_level_5_name"):
            print(f"  {tier:14s} n={len(group):4d}  mean_err={group['pixel_error'].mean():7.2f}px  "
                  f"within_5px={(group['pixel_error']<=5).mean()*100:5.1f}%")

    if "generation_mode" in out_df.columns:
        print("\n--- Accuracy by generation_mode (the real fault line -- see matching.py notes) ---")
        for mode, group in out_df.groupby("generation_mode"):
            print(f"  {mode:14s} n={len(group):4d}  mean_err={group['pixel_error'].mean():7.2f}px  "
                  f"within_5px={(group['pixel_error']<=5).mean()*100:5.1f}%")

    if {"reference_centered_on_landmark", "straddles_boundary"}.issubset(set(pd.read_csv(
            os.path.join(args.dataset, "ground_truth.csv")).columns)):
        gt_df = pd.read_csv(os.path.join(args.dataset, "ground_truth.csv"))[
            ["sample_id", "reference_centered_on_landmark", "straddles_boundary"]]
        merged = out_df.merge(gt_df, on="sample_id", how="left")
        merged["has_context"] = (merged["reference_centered_on_landmark"].astype(bool) |
                                  merged["straddles_boundary"].astype(bool))
        print("\n--- Accuracy by disambiguating context (THE key split -- read this before quoting "
              "an overall pixel-error number) ---")
        for has_ctx, group in merged.groupby("has_context"):
            label = "has landmark/street context" if has_ctx else "pure periodic, NO context"
            print(f"  {label:32s} n={len(group):4d}  mean_err={group['pixel_error'].mean():7.2f}px  "
                  f"median_err={group['pixel_error'].median():6.2f}px  "
                  f"within_5px={(group['pixel_error']<=5).mean()*100:5.1f}%  "
                  f"within_2px={(group['pixel_error']<=2).mean()*100:5.1f}%")
        print("  NOTE: the no-context bucket is not a bug -- for a genuinely, perfectly periodic "
              "region with no unique feature, several locations are visually IDENTICAL matches to "
              "the reference. No local-matching algorithm can reliably localize those to <5px. "
              "Report both numbers separately; don't blend them into one headline figure.")

    worst = out_df.loc[out_df["pixel_error"].idxmax()]
    print(f"\nWorst case: {worst['sample_id']}  error={worst['pixel_error']:.2f}px")
    save_failure_case(args.dataset, worst)

    print(f"\nManifest saved -> {args.out_csv}")


def save_failure_case(dataset_dir, worst_row):
    search = cv2.imread(os.path.join(dataset_dir, worst_row["search_file"]))
    gt = (int(worst_row["gt_x"]), int(worst_row["gt_y"]))
    pred = (int(worst_row["pred_x"]), int(worst_row["pred_y"]))
    cv2.drawMarker(search, gt, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.drawMarker(search, pred, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    out_path = f"failure_case_{worst_row['sample_id']}.png"   # relative, not absolute
    cv2.imwrite(out_path, search)
    print(f"Failure case saved -> {out_path}  (green=ground truth, red=prediction)")


if __name__ == "__main__":
    main()
