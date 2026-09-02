# Drift-Sense Phase 2 -- Registration under Unknown Pose

Applied Materials Semicon India Hackathon 2026, Phase 2 submission.

## Approach

**Classical computer vision only** -- multi-scale, multi-rotation normalized
cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`) over the
disclosed pose space (zoom `[8,12]x`, rotation `+-5deg`), with sub-pixel
refinement and a nearest-to-centre tie-break among near-tied top
candidates. No learned/CNN component in the graded path.

This is a deliberate choice, not an omission. A Siamese CNN reranker was
trained and evaluated twice (Phase 1 and Phase 2, two architectures, seeded
training, LR scheduling, early stopping) and scored **worse than classical
CV both times** -- see `failure_analysis.md` for the full comparison. The
CNN training/eval code is kept in `training/` as documented ablation
evidence, but is not wired into `register.py`.

This also matches the addendum's rule against a "method materially
different from your Phase 1 declared approach" -- Phase 1's own
verified-best method was CV-only, and Phase 2 extends that same method to
the wider pose space rather than replacing it.

## Folder structure

```
register.py              REQUIRED entry point -- the only thing the
                          organizers actually run for grading
matching.py               candidate generation (multi-scale/rotation NCC
                          + sub-pixel refinement)
generate_dataset.py       documented synthetic dataset generator
requirements.txt          pip-freeze-style pinned deps (CPU-only torch)
failure_analysis.md       CV-vs-CNN comparison, known limitations, runtime

eval/
  evaluate.py              evaluation harness -- CV-only by default,
                            uses the EXACT SAME selection logic as
                            register.py so numbers are trustworthy
  make_baseline_report.py  formats a manifest into the organizer's own
                            baseline_calibration.txt table style

training/                  ABLATION ONLY -- not used by register.py
  model_v2.py               ImprovedEmbedNet architecture
  train_reranker.py         Siamese triplet-loss training
  mine_hard_negatives.py    hard-negative mining for training
  calibrate_rejection.py    calibrates the found/score threshold that
                             register.py's FOUND_THRESHOLD is set from
```

Not included in this download (regenerate locally, see commands below):
`out_setA/ out_setB/ out_setC/` (datasets), `model/*.pt` (CNN weights,
ablation only), `hard_negatives.csv`, `test_split.csv`, `eval/results/`.

## Running register.py (the graded command)

```bash
python register.py --input pairs.csv --output predictions.csv
```

`pairs.csv` needs a pair id column plus a reference-path and search-path
column. Either of these header conventions is accepted:

```
pair_id, reference_path, search_path      # organizer sample format
pair_id, reference_file, search_file      # our own generate_dataset.py format
```

`predictions.csv` is written with exactly: `pair_id, x, y, theta, scale,
found, score`. When `found=0`, `x/y/theta/scale` are all written as `0`,
per the addendum's contract.

No GPU, no network, no torch import anywhere in `register.py` or
`matching.py` -- only `numpy`, `opencv-python`, `pandas`.

### Found/rejection threshold

`register.py`'s `FOUND_THRESHOLD = 0.335` was calibrated with
`training/calibrate_rejection.py` against our own generated Set A/B
(present) vs Set C (absent) data: best threshold 0.335, F1 0.916 on that
split. Recalibrate against different data with:

```bash
python training/calibrate_rejection.py --present out_setA out_setB --absent out_setC
```
then update the `FOUND_THRESHOLD` constant in `register.py` (or pass
`--found-threshold <value>` at call time).

### Runtime budget

Addendum requires median <= 5s/pair, hard timeout 20s/pair. On our own
development machine (not the reference machine), median runtime hovered
close to this budget (~3.8-5.4s depending on run); see `failure_analysis.md`
for the honest discussion of this risk and what was/wasn't tried to
address it.

## Step-by-step: regenerate data, calibrate, evaluate

```bash
# 1. install deps (CPU-only torch only needed for the ablation eval mode)
pip install -r requirements.txt
pip install torch==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. generate evaluation data
python generate_dataset.py --architecture dram --num_pairs 300 --set A --output_dir out_setA
python generate_dataset.py --architecture dram --num_pairs 300 --set B --output_dir out_setB
python generate_dataset.py --architecture dram --num_pairs 150 --set C --output_dir out_setC

# 3. calibrate the found/rejection threshold
python training/calibrate_rejection.py --present out_setA out_setB --absent out_setC
# -> update register.py's FOUND_THRESHOLD with the printed value

# 4. sanity-check register.py's I/O contract end to end
python register.py --input out_setA/ground_truth.csv --output eval/results/predictions_smoke.csv

# 5. full evaluation (CV-only, matches register.py exactly)
python eval/evaluate.py --datasets out_setA out_setB out_setC --split all \
    --found-threshold 0.335 --out_dir eval/results

# 6. format a baseline_calibration.txt-style report
python eval/make_baseline_report.py --manifest eval/results/manifest_cv.csv \
    --threshold 0.335 --out eval/results/baseline_report.txt
```

Step 4 works directly because our own `ground_truth.csv` files have
`reference_file`/`search_file` columns that `register.py` recognizes --
no separate `pairs.csv` needed for local sanity checks.

### Optional: reproduce the CNN ablation (not used in the final submission)

```bash
python training/mine_hard_negatives.py --datasets out_setA out_setB --out hard_negatives.csv
python training/train_reranker.py --datasets out_setA out_setB --hard_neg hard_negatives.csv \
    --epochs 40 --model tiny --seed 42 --out model/reranker_tiny.pt
python eval/evaluate.py --datasets out_setA out_setB out_setC \
    --model tiny --reranker model/reranker_tiny_weights.pt --split test \
    --out_dir eval/results
```

## Generator citations

`generate_dataset.py --output_dir <dir>` also writes `<dir>/citations.json`,
mapping each synthetic pattern family to the literature context that
motivated it (IRDS/ITRS roadmaps, DRAM/FinFET process papers, patents).
These are cited as technology/layout context for the generation strategy,
not as sources of literal numeric generation parameters.
