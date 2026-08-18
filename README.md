# SEM Navigation Error Recovery — Wafer Site Localization


**Demo Video:** 
--
**test using frontend** - https://www.loom.com/share/250bc63399fd41418f3679019de84057
--
**tested on 30 samples video link** - https://drive.google.com/file/d/1OfFrgBS9DLr80Lq26Uc-kUlDCYQcnnfX/view
--
**Dataset:** `https://github.com/nehayadav827/Applied-Materials-Semicon-2026/tree/main/Eval_Dataset`

**Unique images** https://docs.google.com/document/d/19BzJttnVhZkfE4RlggFAp4VYJQfLYAneaWQp9SvFHdA/edit?usp=sharing

**Citation and Reference** https://docs.google.com/document/d/1stQ7oAZ0lftw6mrP_gJRJzDH5o9SCctmOhUHGbZmoew/edit?usp=sharing
---

## Overview

Semiconductor inspection systems must repeatedly return to the same wafer location, but small navigation errors (thermal drift, vibration, mechanical inaccuracy) can shift the actual inspection site. Because DRAM, FinFET, and interconnect layouts are highly repetitive, a wrong location can look almost identical to the correct one.

**Task:** given a small high-magnification **Reference Image** and a larger, lower-magnification **Search Image** (~10× scale difference) of the same layout, find where the reference occurs in the search image and return the center `(x, y)`.

---

## Approach

**Pipeline:** `reference.png + search.png → NCC matching across 45 scale/rotation combos → top-6 peaks per map, non-max suppressed → global merge + rank by score → sub-pixel refine best candidate → (x, y)`

`matching.py` sweeps 9 scales × 5 rotation angles (45 combinations), running NCC template matching at each. From each correlation map it keeps up to 6 local peaks, suppresses near-duplicates, then merges and ranks all peaks globally. The top candidate is refined to sub-pixel accuracy via parabolic interpolation.

**Decision Rule:** pick the top-scoring candidate from the sweep as the final prediction. An optional CNN-reranker mode can instead pick by smallest embedding distance to the reference, breaking ties within `0.02` by choosing the candidate closest to the search image's center — but plain CV top-1 matching performs better and is the shipped default.

---

## Dataset

Synthetic reference/search pairs generated to reproduce realistic wafer-navigation challenges: highly repetitive structures, multiple pattern families (DRAM, FinFET, interconnect), duplicate "twin" regions, and SEM-style degradation (beam blur, drift, shot noise, charging streaks, edge brightening).

```bash
python generate_dataset.py --architecture dram --num_pairs 30 --output_dir ./synthetic_sem_dataset
```
(Omit `--architecture` for a mixed-pattern dataset by default.)

Each sample includes `reference.png`, `search.png`, `visualization.png`, and `metadata.json`, with ground-truth coordinates in `ground_truth.csv`.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# 1. Precompute hard negatives (only needed if training the optional reranker)
python precompute_hard_negatives.py --dataset ./synthetic_sem_dataset --out hard_negatives.csv

# 2. Localize a single reference/search pair
python localize.py --reference <reference_image_path> --search <search_image_path>
# Add --use-cnn --reranker reranker.pt to enable the learned reranker; default mode is pure CV.

# 3. Evaluate on the full dataset
python evaluate.py --dataset ./synthetic_sem_dataset --reranker reranker.pt --split test --out_csv evaluation_manifest.csv
# Use --no-cnn to benchmark the raw CV top-1 candidate without the reranker.

# 4. Launch the web demo
cd frontend
uvicorn api:app --reload --port 8000
```
Open `http://localhost:8000` to upload a reference/search pair and view the localization result.

---

## Results — 30-Test-Case Evaluation

**Hardware:** Intel CPU | Python 3.12 | OpenCV 4.11

| Metric | Value |
|---|---:|
| Within 5 px | 80.0 % |
| Within 4 px | 73.3 % |
| Within 2 px | 63.3 % |
| Within 1 px | 26.7 % |
| Median error | 1.80 px |
| Mean error | 36.04 px (skewed by a few large failures) |
| Runtime | 0.908 s / 1000×1000 image pair on CPU |

**Success case — `sample_0203`:** ground truth (182.20, 309.20), prediction (181.95, 309.16), error 0.25 px. Demonstrates accurate sub-pixel localization when the target has enough distinguishing structure.

**Honest failure — `sample_1181`:** ground truth (77.40, 661.90), prediction (740.32, 410.50), error 708.99 px. Caused by a highly repetitive/ambiguous region where the matcher locked onto a visually similar but incorrect location — the fundamental limit of periodic semiconductor layouts when no unique context is present.

**Key takeaway:** the method achieves high-precision, sub-pixel localization on distinguishable regions (1.80 px median error), while highly repetitive regions remain the principal failure mode.

---

## Limitations

- **Synthetic-to-real gap:** the dataset is synthetic; real SEM imagery may include effects not fully captured by the generator.
- **Fundamental periodic ambiguity:** perfectly periodic regions with no unique context can be genuinely indistinguishable from the reference.
- **Candidate-generation dependency:** the reranker can only choose among candidates already proposed by the CV stage — if the correct location isn't proposed, it can't be recovered.

---

## Project Structure

```
├── generate_dataset.py       # Synthetic dataset generator
├── matching.py                # Candidate generation (CV/NCC matching)
├── localize.py                # Main localization entry point
├── precompute_hard_negatives.py
├── train_reranker.py          # Optional CNN reranker training
├── evaluate.py                # Batch evaluation
├── reranker.pt                # Trained reranker weights (optional mode)
├── Eval_Dataset/               # Evaluation sample pairs + ground truth
└── frontend/                   # FastAPI web demo
    ├── api.py
    └── static/index.html
```
