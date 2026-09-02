"""
eval/make_baseline_report.py -- reformats an eval/evaluate.py manifest CSV
into the same table style as the organizer's own worked-reference
baseline_calibration.txt (see the Phase 2 sample-pairs README), so our own
baseline report is directly comparable in shape to theirs:

    id    set  pres peak   err_px   z_hat/z        th_hat/th      credit
    ...
    --- calibration ---
    present peaks : min=... max=...
    absent  peaks : min=... max=...
    separation gap: ...  (positive = rejectable by threshold)
    rejection @ thr=...: TP=.. FP=.. FN=.. precision=.. recall=.. F1=..
    Set A: mean credit=... median err=...px
    ...
    overall mean credit (present pairs): ...

Usage:
    python eval/make_baseline_report.py --manifest eval/results/manifest_cv.csv \
        --threshold 0.335 --out eval/results/baseline_report.txt
"""

import argparse

import numpy as np
import pandas as pd


def infer_set_label(sample_uid, true_found, generation_mode):
    """We don't carry organizer-style A/B/C/D set labels through our own
    ground_truth.csv, so approximate: absent -> C, everything else is
    labelled by whether it's degraded (loosely: 'B' if difficulty/severity
    fields suggest degradation) vs nominal ('A'). This is a best-effort
    label for report readability only -- it does not affect scoring."""
    if not true_found:
        return "C"
    return "A/B"  # our present pairs mix nominal+degraded; see is_degraded_pair in ground_truth.csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold", type=float, default=0.335)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    lines = []
    lines.append(f"{'id':14s}{'set':5s}{'pres':5s}{'peak':7s}{'err_px':9s}"
                 f"{'z_hat/z':16s}{'th_hat/th':16s}{'credit':7s}")

    for _, row in df.iterrows():
        set_label = infer_set_label(row["sample_uid"], row["true_found"], row.get("generation_mode", ""))
        pres = int(row["true_found"])
        peak = row["score"]
        if pres:
            err = row["pixel_error"]
            credit = row["loc_credit"] if pd.notna(row["loc_credit"]) else 0.0
            z_str = "n/a"  # z_hat/z requires GT_scale + pred_scale, not both carried in this manifest shape
            th_str = "n/a"
        else:
            err = float("nan")
            credit = ""
            z_str = "-"
            th_str = "-"

        lines.append(f"{str(row['sample_uid']):14s}{set_label:5s}{pres:<5d}{peak:<7.3f}"
                     f"{err if pres else float('nan'):<9.2f}{z_str:16s}{th_str:16s}{str(credit):7s}")

    lines.append("")
    lines.append("--- calibration ---")

    present = df[df["true_found"] == 1]
    absent = df[df["true_found"] == 0]

    if len(present):
        lines.append(f"present peaks : min={present['score'].min():.3f} max={present['score'].max():.3f}")
    if len(absent):
        lines.append(f"absent  peaks : min={absent['score'].min():.3f} max={absent['score'].max():.3f}")

    if len(present) and len(absent):
        gap = present["score"].min() - absent["score"].max()
        lines.append(f"separation gap: {gap:.3f}  (positive = rejectable by threshold)")

        y_true = df["true_found"].astype(int).values
        y_pred = (df["score"] >= args.threshold).astype(int).values
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        lines.append(f"rejection @ thr={args.threshold}: TP={tp} FP={fp} FN={fn} "
                     f"precision={precision:.2f} recall={recall:.2f} F1={f1:.3f}")

    if len(present):
        errs = present["pixel_error"].astype(float)
        credits = present["loc_credit"].astype(float)
        lines.append(f"Overall (present): mean credit={credits.mean():.3f} median err={errs.median():.2f}px")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Report written -> {args.out}")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
