"""
Final evaluation script for Drift-Sense.

Produces:
- Evaluation manifest CSV
- 5px / 4px / 2px / 1px pass rates
- Mean / median / worst pixel error
- Runtime / hardware report
- Accuracy by difficulty
- Accuracy by generation mode
- Accuracy by disambiguating context
- Worst-case failure image
- 5-pixel threshold confusion matrix

By default this scores ONLY the held-out test split written by
train_reranker.py (dataset/test_split.csv).

For an independent evaluation folder such as Eval_Dataset,
use:

    python evaluate.py --dataset ./Eval_Dataset --split all --limit 30

For CV-only evaluation:

    python evaluate.py --dataset ./Eval_Dataset --no-cnn --split all --limit 30

Coordinate convention:
- Origin (0,0) is the TOP-LEFT of the search image
- x increases right
- y increases down
"""

import argparse
import os
import platform
import time

import cv2
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from matching import generate_candidates
from training.train_reranker import TinyEmbedNet, PATCH_SIZE


# ============================================================
# CONFIGURATION
# ============================================================

CENTER_TIE_MARGIN = 0.02


# ============================================================
# IMAGE -> TENSOR HELPERS
# ============================================================

def to_patch_tensor(img_patch):
    """
    Convert a candidate search patch into the tensor format
    expected by the reranker.
    """

    p = cv2.resize(
        img_patch,
        (PATCH_SIZE, PATCH_SIZE)
    )

    return (
        torch.from_numpy(p)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )


def to_ref_tensor(reference, scale):
    """
    Pre-downsample the reference according to the candidate scale
    before resizing to PATCH_SIZE.

    This keeps the reference and candidate patch at comparable
    detail levels.
    """

    h, w = reference.shape[:2]

    pre_w = max(
        8,
        int(round(w / scale))
    )

    pre_h = max(
        8,
        int(round(h / scale))
    )

    p = cv2.resize(
        reference,
        (pre_w, pre_h),
        interpolation=cv2.INTER_AREA
    )

    p = cv2.resize(
        p,
        (PATCH_SIZE, PATCH_SIZE)
    )

    return (
        torch.from_numpy(p)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )


# ============================================================
# LOCALIZATION
# ============================================================

def localize_one(reference, search, model):
    """
    Localize the reference structure inside the search image.

    If model is None:
        Uses CV-only candidate ranking.

    If model is provided:
        Uses CNN reranking followed by the closest-to-centre rule.
    """

    h, w = search.shape[:2]

    search_center = (
        w / 2.0,
        h / 2.0
    )

    # Generate CV candidates.
    candidates = generate_candidates(
        reference,
        search,
        top_k=30
    )

    # Degenerate fallback.
    if not candidates:
        return (
            w / 2.0,
            h / 2.0
        )

    # --------------------------------------------------------
    # CV-ONLY MODE
    # --------------------------------------------------------

    if model is None:

        return (
            candidates[0][1],
            candidates[0][2]
        )

    # --------------------------------------------------------
    # CNN RERANKING
    # --------------------------------------------------------

    with torch.no_grad():

        ref_emb_cache = {}

        def ref_embedding(scale):

            if scale not in ref_emb_cache:

                ref_emb_cache[scale] = model(
                    to_ref_tensor(
                        reference,
                        scale
                    )
                )

            return ref_emb_cache[scale]

        scored = []

        for (
            score,
            cx,
            cy,
            scale,
            angle
        ) in candidates:

            pw = max(
                int(reference.shape[1] / scale),
                8
            )

            ph = max(
                int(reference.shape[0] / scale),
                8
            )

            x0 = max(
                0,
                int(cx - pw / 2)
            )

            y0 = max(
                0,
                int(cy - ph / 2)
            )

            patch = search[
                y0:y0 + ph,
                x0:x0 + pw
            ]

            if patch.size == 0:
                continue

            emb = model(
                to_patch_tensor(patch)
            )

            dist = (
                ref_embedding(scale) - emb
            ).pow(2).sum().item()

            scored.append(
                (
                    dist,
                    cx,
                    cy
                )
            )

    # Fallback if CNN could not score anything.
    if not scored:
        return search_center

    # Best embedding distance first.
    scored.sort(
        key=lambda s: s[0]
    )

    best_dist = scored[0][0]

    # --------------------------------------------------------
    # CLOSEST-TO-CENTRE RULE
    # --------------------------------------------------------

    valid = [
        s
        for s in scored
        if s[0] <= best_dist + CENTER_TIE_MARGIN
    ]

    valid.sort(
        key=lambda s:
        (
            (s[1] - search_center[0]) ** 2
            +
            (s[2] - search_center[1]) ** 2
        )
    )

    _, cx, cy = valid[0]

    return cx, cy


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(out_df, out_csv):
    """
    Save a 5-pixel threshold confusion matrix.

    Pass definition:
        pixel_error <= 5 px

    Fail definition:
        pixel_error > 5 px

    This matrix visualizes the pass/fail distribution at the
    required 5-pixel acceptance threshold.

    Note:
        There is no separate classifier prediction here.
        The same 5-pixel criterion defines the pass/fail result,
        so this is a threshold summary rather than an independent
        classification-model confusion matrix.
    """

    if out_df.empty:
        print(
            "No evaluation samples available "
            "for confusion matrix."
        )
        return

    # --------------------------------------------------------
    # 5-PIXEL PASS/FAIL LABEL
    # --------------------------------------------------------

    labels = (
        out_df["pixel_error"] <= 5
    ).astype(int).values

    # For the threshold visualization, actual and predicted
    # labels are the same pass/fail decision.
    y_true = labels
    y_pred = labels

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print(
        "\n--- Confusion Matrix "
        "(5 px localization threshold) ---"
    )

    print(
        "                 Predicted Fail   Predicted Pass"
    )

    print(
        f"Actual Fail      {cm[0, 0]:15d}   "
        f"{cm[0, 1]:14d}"
    )

    print(
        f"Actual Pass      {cm[1, 0]:15d}   "
        f"{cm[1, 1]:14d}"
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    out_prefix = os.path.splitext(
        out_csv
    )[0]

    cm_csv_path = (
        out_prefix
        + "_confusion_matrix.csv"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Fail",
            "Actual Pass"
        ],
        columns=[
            "Predicted Fail",
            "Predicted Pass"
        ]
    )

    cm_df.to_csv(
        cm_csv_path
    )

    print(
        f"Confusion matrix CSV saved -> "
        f"{cm_csv_path}"
    )

    # --------------------------------------------------------
    # SAVE PNG
    # --------------------------------------------------------

    cm_png_path = (
        out_prefix
        + "_confusion_matrix.png"
    )

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    im = ax.imshow(
        cm,
        cmap="viridis"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Fail",
            "Pass"
        ]
    )

    ax.set_yticklabels(
        [
            "Fail",
            "Pass"
        ]
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        "Confusion Matrix (5 px threshold)"
    )

    # Write numbers inside the cells.
    for r in range(2):

        for c in range(2):

            ax.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(
        im,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        cm_png_path,
        dpi=200
    )

    plt.close(fig)

    print(
        f"Confusion matrix plot saved -> "
        f"{cm_png_path}"
    )


# ============================================================
# FAILURE CASE
# ============================================================

def save_failure_case(
    dataset_dir,
    worst_row
):
    """
    Save the worst localization result.

    Green marker = ground truth
    Red marker   = prediction
    """

    search_path = os.path.join(
        dataset_dir,
        worst_row["search_file"]
    )

    search = cv2.imread(
        search_path
    )

    if search is None:

        print(
            f"WARNING: Could not read failure-case image: "
            f"{search_path}"
        )

        return

    gt = (
        int(worst_row["gt_x"]),
        int(worst_row["gt_y"])
    )

    pred = (
        int(worst_row["pred_x"]),
        int(worst_row["pred_y"])
    )

    # Ground truth = green.
    cv2.drawMarker(
        search,
        gt,
        (0, 255, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    # Prediction = red.
    cv2.drawMarker(
        search,
        pred,
        (0, 0, 255),
        cv2.MARKER_TILTED_CROSS,
        20,
        2
    )

    out_path = (
        f"failure_case_"
        f"{worst_row['sample_id']}.png"
    )

    cv2.imwrite(
        out_path,
        search
    )

    print(
        f"Failure case saved -> {out_path} "
        f"(green=ground truth, red=prediction)"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="./synthetic_sem_dataset"
    )

    parser.add_argument(
        "--reranker",
        default="reranker.pt"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "test",
            "all"
        ],
        help=(
            "'test' scores only the held-out split written "
            "by train_reranker.py; "
            "'all' scores the whole dataset "
            "(sanity/ablation only)."
        )
    )

    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help=(
            "Skip the reranker and use the raw CV "
            "top-1 candidate."
        )
    )

    parser.add_argument(
        "--out_csv",
        default="evaluation_manifest.csv"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress per-sample pixel-error output."
        )
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD RERANKER
    # ========================================================

    model = None

    if not args.no_cnn:

        print(
            f"Loading reranker -> "
            f"{args.reranker}"
        )

        model = TinyEmbedNet(
            embed_dim=128
        )

        model.load_state_dict(
            torch.load(
                args.reranker,
                map_location="cpu"
            )
        )

        model.eval()

    else:

        print(
            "CNN reranker disabled "
            "(CV-only ablation mode)"
        )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    ground_truth_path = os.path.join(
        args.dataset,
        "ground_truth.csv"
    )

    if not os.path.exists(
        ground_truth_path
    ):

        raise FileNotFoundError(
            f"Ground-truth file not found:\n"
            f"{ground_truth_path}"
        )

    df = pd.read_csv(
        ground_truth_path
    )

    # ========================================================
    # SELECT SPLIT
    # ========================================================

    if args.split == "test":

        split_path = os.path.join(
            args.dataset,
            "test_split.csv"
        )

        if not os.path.exists(
            split_path
        ):

            raise FileNotFoundError(
                f"{split_path} not found.\n"
                f"Run train_reranker.py first, "
                f"or use --split all."
            )

        test_ids = pd.read_csv(
            split_path
        )["sample_id"]

        df = df[
            df["sample_id"].isin(
                test_ids
            )
        ].reset_index(
            drop=True
        )

        print(
            f"Scoring held-out TEST split only: "
            f"{len(df)} samples"
        )

    else:

        print(
            
        )

    # ========================================================
    # LIMIT
    # ========================================================

    if args.limit is not None:

        df = df.head(
            args.limit
        ).reset_index(
            drop=True
        )

        print(
            f"Limit applied: "
            f"{len(df)} samples"
        )

    if df.empty:

        raise RuntimeError(
            "No samples selected for evaluation."
        )

    # ========================================================
    # HARDWARE
    # ========================================================

    print(
        "Hardware:",
        platform.processor(),
        "| device=cpu",
        "| python=" + platform.python_version(),
        "| torch=" + torch.__version__,
        "| opencv=" + cv2.__version__
    )

    print(
        "Timing method: wall-clock "
        "time.perf_counter() around "
        "candidate generation + CNN rerank "
        "+ centre tie-break, per sample.\n"
    )

    # ========================================================
    # EVALUATION LOOP
    # ========================================================

    rows = []
    times = []

    for i, row in df.iterrows():

        ref_path = os.path.join(
            args.dataset,
            row["reference_file"]
        )

        search_path = os.path.join(
            args.dataset,
            row["search_file"]
        )

        ref = cv2.imread(
            ref_path,
            cv2.IMREAD_GRAYSCALE
        )

        search = cv2.imread(
            search_path,
            cv2.IMREAD_GRAYSCALE
        )

        if ref is None:

            raise FileNotFoundError(
                f"Could not read reference image:\n"
                f"{ref_path}"
            )

        if search is None:

            raise FileNotFoundError(
                f"Could not read search image:\n"
                f"{search_path}"
            )

        # ----------------------------------------------------
        # LOCALIZATION
        # ----------------------------------------------------

        t0 = time.perf_counter()

        pred_x, pred_y = localize_one(
            ref,
            search,
            model
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        times.append(
            elapsed
        )

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        gt_x = float(
            row["GT_X"]
        )

        gt_y = float(
            row["GT_Y"]
        )

        # ----------------------------------------------------
        # EUCLIDEAN PIXEL ERROR
        # ----------------------------------------------------

        err = float(
            np.sqrt(
                (pred_x - gt_x) ** 2
                +
                (pred_y - gt_y) ** 2
            )
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        rows.append(
            {
                "sample_id":
                    row["sample_id"],

                "reference_file":
                    row["reference_file"],

                "search_file":
                    row["search_file"],

                "difficulty_level_5_name":
                    row.get(
                        "difficulty_level_5_name",
                        ""
                    ),

                "generation_mode":
                    row.get(
                        "generation_mode",
                        ""
                    ),

                "gt_x":
                    gt_x,

                "gt_y":
                    gt_y,

                "pred_x":
                    round(
                        pred_x,
                        2
                    ),

                "pred_y":
                    round(
                        pred_y,
                        2
                    ),

                "pixel_error":
                    round(
                        err,
                        2
                    ),

                "pass_5px":
                    err <= 5,

                "pass_4px":
                    err <= 4,

                "pass_2px":
                    err <= 2,

                "pass_1px":
                    err <= 1,

                "runtime_s":
                    round(
                        elapsed,
                        4
                    ),
            }
        )

        # ----------------------------------------------------
        # PRINT SAMPLE RESULT
        # ----------------------------------------------------

        if not args.quiet:

            print(
                f"[{i + 1:4d}/{len(df)}] "
                f"{str(row['sample_id']):16s} "
                f"pred=({pred_x:7.2f},"
                f"{pred_y:7.2f})  "
                f"gt=({gt_x:7.2f},"
                f"{gt_y:7.2f})  "
                f"error={err:8.2f}px  "
                f"{elapsed:.3f}s"
            )

        # Running mean every 50 samples.
        if (
            (i + 1) % 50 == 0
        ):

            running_errors = [
                r["pixel_error"]
                for r in rows
            ]

            print(
                f"  -- running "
                f"mean_err="
                f"{np.mean(running_errors):.2f}px "
                f"over {i + 1} samples --"
            )

    # ========================================================
    # OUTPUT DATAFRAME
    # ========================================================

    out_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    out_df.to_csv(
        args.out_csv,
        index=False
    )

    print(
        f"\nManifest saved -> "
        f"{args.out_csv}"
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    errors = (
        out_df["pixel_error"]
        .values
    )

    print(
        "\n--- FINAL RESULTS ---"
    )

    print(
        f"Split: {args.split}"
    )

    print(
        f"Samples evaluated: "
        f"{len(out_df)}"
    )

    print(
        f"Mean error:   "
        f"{errors.mean():.2f} px"
    )

    print(
        f"Median error: "
        f"{np.median(errors):.2f} px"
    )

    # Required threshold results.
    for thresh in [
        5,
        4,
        2,
        1
    ]:

        pct = (
            errors <= thresh
        ).mean() * 100

        print(
            f"  within {thresh}px: "
            f"{pct:.1f}%"
        )

    print(
        f"Avg runtime: "
        f"{np.mean(times):.3f}s/sample "
        f"({sum(times):.1f}s total compute)"
    )

    # ========================================================
    # ACCURACY BY DIFFICULTY
    # ========================================================

    if (
        "difficulty_level_5_name"
        in out_df.columns
    ):

        print(
            "\n--- Accuracy by difficulty tier ---"
        )

        for (
            tier,
            group
        ) in out_df.groupby(
            "difficulty_level_5_name"
        ):

            print(
                f"  {str(tier):14s} "
                f"n={len(group):4d}  "
                f"mean_err="
                f"{group['pixel_error'].mean():7.2f}px  "
                f"within_5px="
                f"{(group['pixel_error'] <= 5).mean() * 100:5.1f}%"
            )

    # ========================================================
    # ACCURACY BY GENERATION MODE
    # ========================================================

    if (
        "generation_mode"
        in out_df.columns
    ):

        print(
            "\n--- Accuracy by generation_mode ---"
        )

        for (
            mode,
            group
        ) in out_df.groupby(
            "generation_mode"
        ):

            print(
                f"  {str(mode):14s} "
                f"n={len(group):4d}  "
                f"mean_err="
                f"{group['pixel_error'].mean():7.2f}px  "
                f"within_5px="
                f"{(group['pixel_error'] <= 5).mean() * 100:5.1f}%"
            )

    # ========================================================
    # ACCURACY BY DISAMBIGUATING CONTEXT
    # ========================================================

    ground_truth_columns = pd.read_csv(
        ground_truth_path,
        nrows=0
    ).columns

    if {
        "reference_centered_on_landmark",
        "straddles_boundary"
    }.issubset(
        set(ground_truth_columns)
    ):

        gt_df = pd.read_csv(
            ground_truth_path
        )[
            [
                "sample_id",
                "reference_centered_on_landmark",
                "straddles_boundary"
            ]
        ]

        merged = out_df.merge(
            gt_df,
            on="sample_id",
            how="left"
        )

        merged["has_context"] = (
            merged[
                "reference_centered_on_landmark"
            ].fillna(False).astype(bool)
            |
            merged[
                "straddles_boundary"
            ].fillna(False).astype(bool)
        )

        print(
            "\n--- Accuracy by disambiguating context ---"
        )

        for (
            has_ctx,
            group
        ) in merged.groupby(
            "has_context"
        ):

            if has_ctx:

                label = (
                    "has landmark/street context"
                )

            else:

                label = (
                    "pure periodic, NO context"
                )

            print(
                f"  {label:32s} "
                f"n={len(group):4d}  "
                f"mean_err="
                f"{group['pixel_error'].mean():7.2f}px  "
                f"median_err="
                f"{group['pixel_error'].median():6.2f}px  "
                f"within_5px="
                f"{(group['pixel_error'] <= 5).mean() * 100:5.1f}%  "
                f"within_2px="
                f"{(group['pixel_error'] <= 2).mean() * 100:5.1f}%"
            )

        print(
            "  NOTE: the no-context bucket can contain "
            "genuinely periodic regions where multiple "
            "locations are visually identical."
        )

    # ========================================================
    # WORST CASE
    # ========================================================

    worst = out_df.loc[
        out_df["pixel_error"].idxmax()
    ]

    print(
        f"\nWorst case: "
        f"{worst['sample_id']}  "
        f"error="
        f"{worst['pixel_error']:.2f}px"
    )

    save_failure_case(
        args.dataset,
        worst
    )

    # ========================================================
    # CONFUSION MATRIX ONLY
    # ========================================================

    save_confusion_matrix(
        out_df,
        args.out_csv
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n--- EVALUATION COMPLETE ---"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
