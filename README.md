# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

Locates a high-resolution (100x) reference-image crop inside a wider,
noisier, lower-resolution (10x) search image, for the Applied Materials
Drift-Sense hackathon problem (SEMICON India 2026).

## Repository contents

| File | Purpose |
|---|---|
| `generate_dataset.py` | Standalone dataset generator. Produces (reference, search) image pairs with recorded ground-truth center coordinates. |
| `localize.py` | **Standalone inference script.** Takes a reference image path and a search image path, prints the predicted center `(x, y)`. This is the script meant to be run directly on new test pairs. |
| `matching.py` | Shared classical CV matching logic (multi-scale/rotation template matching, sub-pixel refinement). Imported by every other script here. |
| `train_reranker.py` | Trains an optional Siamese CNN reranker on a generated dataset. Produces `reranker.pt`. |
| `precompute_hard_negatives.py` | One-time preprocessing step required before `train_reranker.py`. |
| `evaluate.py` | Batch validation script: runs `localize.py`'s logic over an entire generated dataset's held-out test split and reports pixel-error pass rates, runtime, and a failure case. |
| `reranker.pt` | Trained CNN reranker weights (optional -- see "Which mode should I use" below). |
| `requirements.txt` | Python dependencies. |
| `references.md` | Citations justifying the synthetic pattern styles and noise model. |

## Setup

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`localize.py` in its default mode (recommended -- see below) does **not**
require `torch` at all. If you only want to run inference and skip
training/the CNN-hybrid mode, you can omit `torch` from the install.

## Quick start: generate a sample pair and localize it

```bash
# 1. Generate a small sample dataset (3 pairs, mixed DRAM+FinFET styles)
python generate_dataset.py --style mixed --num_pairs 3 --output_dir ./sample_data

# 2. Run the inference script on one generated pair
python localize.py --reference ./sample_data/sample_0000/reference.png \
                    --search ./sample_data/sample_0000/search.png
```

Expected output: a single line on stdout, e.g.

```
512.340,487.910
```

That's the predicted `(x, y)` center of the reference pattern within the
search image, in search-image pixel coordinates. All other messages
(mode, runtime) print to stderr, so stdout always contains exactly the
coordinate line -- see "Inference script output contract" below.

You can compare against ground truth for that sample:

```bash
python -c "import json; m=json.load(open('sample_data/sample_0000/metadata.json')); print(m['GT_X'], m['GT_Y'])"
```

## `generate_dataset.py` -- dataset generator

```bash
python generate_dataset.py --style {dram,finfet,mixed} --num_pairs N --output_dir DIR [--seed S]
```

| Argument | Meaning |
|---|---|
| `--style` | `dram` restricts every generated pattern to DRAM-family structures (dot/capsule arrays, wavy bitlines, layered word/bit-line cells). `finfet` restricts to FinFET/logic-family structures (fin-line grids, gate/via bands, standard-cell tracks). `mixed` (default) uses the full original diversity across both plus additional interconnect/logic pattern families -- every accuracy number in this repo's history was measured in `mixed` mode. |
| `--num_pairs` | Number of (reference, search) pairs to generate. |
| `--output_dir` | Output directory. Created if it doesn't exist. |
| `--seed` | Base random seed (default fixed, for reproducibility). |

Each pair is written to its own subfolder:

```
<output_dir>/sample_0000/
    reference.png        1000x1000 grayscale, 100x magnification
    search.png            1000x1000 grayscale, 10x magnification (nominal 10:1 scale)
    visualization.png     search.png with the ground-truth box/center overlaid
    metadata.json         full per-sample generation parameters + ground truth
<output_dir>/ground_truth.csv      master manifest: paths + GT_X, GT_Y (+ GT bbox) for every pair
<output_dir>/citations.json        per-style citation keys (see references.md)
```

`GT_X` / `GT_Y` in `ground_truth.csv` and each `metadata.json` are the
**true center coordinates of the reference pattern within the search
image**, in search-image pixel coordinates -- the ground truth required
by the checklist.

## `localize.py` -- the inference script

```bash
python localize.py --reference REF.png --search SEARCH.png [--use-cnn] [--json] [--quiet]
```

| Argument | Meaning |
|---|---|
| `--reference` | Path to the reference (100x) image. Required. |
| `--search` | Path to the search (10x) image. Required. |
| `--use-cnn` | Opt into the CNN-reranked hybrid mode (requires `reranker.pt` and `torch`). **Off by default** -- see below for why. |
| `--json` | Print `{"x": ..., "y": ..., "runtime_s": ...}` instead of a bare `x,y` line. |
| `--quiet` | Suppress stderr diagnostics too. |

### Inference script output contract

The **last line printed to stdout** is always exactly `x,y` (two floats,
comma-separated, search-image pixel coordinates, origin top-left) and
nothing else is on that line -- safe for any harness that reads stdout
and parses the final line. All progress/diagnostic messages go to
stderr. Exit code is non-zero (`2`) with a clear stderr message if either
input image path is missing or unreadable.

### Which mode should I use?

**Default (`--use-cnn` not passed): classical CV + sub-pixel refinement.
This is the recommended mode and the one this repo's reported accuracy
numbers use.** We built and trained a Siamese CNN reranker
(`train_reranker.py` / `reranker.pt`, included in this repo for
completeness and because the checklist asks for it), but validated
against the CV-only baseline and found it performed substantially worse
even after fixing a real bug in its reference-image preprocessing (see
commit history / `train_reranker.py` comments for the bug: the model was
comparing the full reference crushed to ~10x lower relative resolution
than the candidate patches it was scored against). On our 500-pair
dataset's 75-sample held-out test split, on samples containing a
landmark or street-boundary feature to disambiguate against:

| Mode | within 5px | within 2px | median error |
|---|---|---|---|
| Classical CV + sub-pixel (default) | 78.6% | 67.9% | 0.83px |
| CV + CNN reranker (`--use-cnn`) | ~25-30% | ~20% | ~60px |

We report this as a documented negative result, not a hidden one --
`--use-cnn` is included for completeness/reproducibility of that
experiment, not because we recommend it.

## Fundamental limitation, honestly documented

Not every reference crop can be localized to sub-5px accuracy, and this
is not a bug. A large fraction of crops sit inside a perfectly periodic
region (dot arrays, line grids) with no unique feature -- several
locations are visually **identical** matches to the reference, which
matches the problem statement's own framing ("if several tiles match,
choose the valid match closest to the centre"). We verified this is an
information-theoretic ceiling, not an implementation gap: pinning the
scale/rotation search to the dataset's exact known 10:1/0° geometry
didn't help, and a Hough-style voting scheme across all candidate peaks
made results *worse*. On our test split:

| Population | within 5px |
|---|---|
| Reference crop contains a landmark or street boundary (~35-40% of samples) | 78.6% |
| Pure periodic texture, no disambiguating feature (~60-65% of samples) | ~4-6% |

`evaluate.py` reports both populations separately every run (see its
"Accuracy by disambiguating context" section) and auto-saves an image of
its single worst case as a concrete, explained failure example.

## Training the reranker (optional)

```bash
python precompute_hard_negatives.py --dataset ./synthetic_sem_dataset --out hard_negatives.csv
python train_reranker.py --dataset ./synthetic_sem_dataset --hard_neg hard_negatives.csv --epochs 30 --out reranker.pt
```

`train_reranker.py` splits the dataset into train/val/test by
**underlying pattern world** (not by raw sample), so no near-duplicate
crop of the same world leaks across the split. It writes
`<dataset>/test_split.csv` with the held-out test `sample_id`s, which
`evaluate.py` reads and scores by default.

## Validation / evaluation

```bash
python evaluate.py --dataset ./synthetic_sem_dataset --no-cnn                       # recommended, matches localize.py's default mode
python evaluate.py --dataset ./synthetic_sem_dataset --reranker reranker.pt         # CNN-hybrid mode, for comparison
```

Reports mean/median/worst pixel error, pass rates at 5/4/2/1px, runtime
per sample with hardware/library versions and timing method stated,
accuracy broken down by difficulty tier / generation mode / disambiguating
context, and saves an image of the single worst-error case.

## Coordinate convention

Origin `(0, 0)` is the **top-left** of the search image; x increases
right, y increases down (standard image-array convention). All
coordinates -- `GT_X`/`GT_Y` in the generator's output, and the `(x, y)`
`localize.py` prints -- use this convention.

## Citations

See `references.md` for the full list of public sources justifying the
synthetic pattern styles and noise model, and how each maps to the
citations used in the presentation slides.

