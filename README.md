# SEM Navigation Error Recovery — Wafer Site Localization

## **Demo Video**
[Demo video](https://drive.google.com/drive/folders/1Y6IRzIQk8RHsYKf-tzXB4Pzm5mz70lB5)

**The ~12-second frontend time includes API, I/O, and UI overhead. The **6.38 s** figure represents the actual registration pipeline runtime.**
## **Dataset**

[Phase 2 Dataset](https://drive.google.com/drive/folders/1Fi7hhRft6D6QYg-Erb6kYf8tx0cTCOqr?usp=sharing)

## **Citation and Reference**

[Citation and Reference — Google Docs](https://docs.google.com/document/d/1stQ7oAZ0lftw6mrP_gJRJzDH5o9SCctmOhUHGbZmoew/edit?usp=sharing)

---

# PHASE 2 — Registration Under Unknown Pose

**Applied Materials Semicon India Hackathon 2026 — Phase 2 Submission**

## Overview

Phase 2 extends the Phase 1 wafer-site localization pipeline to handle **unknown scale (8×–12×), unknown rotation (±5°), reference-presence detection, and RGB optical imagery**, while preserving the same classical computer-vision approach.

The final graded path uses **classical computer vision only**, with no deep-learning dependency.

## Approach

**Classical computer vision only** — multi-scale, multi-rotation normalized cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`) over the disclosed pose space:

-  Scale: **8×–12×** 
-  Rotation: **±5°** 
-  Multiple local NCC candidate peaks 
-  Sub-pixel peak refinement 
-  Nearest-to-centre selection among near-tied candidates 
-  Local scale/rotation refinement 
-  Reference-present / reference-absent decision 

The same standalone registration entry point automatically handles both **grayscale SEM** and **RGB optical** image pairs.

This is a deliberate extension of the Phase 1 approach rather than a replacement. The final graded path remains CV-only.

## Phase 2 Graded Entry Point

The required submission entry point is:

```
python register.py --input pairs.csv --output predictions.csv
```

`pairs.csv` accepts a pair ID together with reference and search image paths.

The prediction output follows the required format:

```
pair_id,x,y,theta,scale,found,score
```

When `found=0`, `x`, `y`, `theta`, and `scale` are written as `0`.

## RGB Extension

Phase 2 extends the registration pipeline to **native 3-channel optical RGB imagery**.

The RGB matcher combines:

- **LAB luminance** 
- **Chromatic information** 
- **Structural gradient information** 

This preserves the geometric matching behaviour of the classical NCC pipeline while adding color as an additional appearance cue.

The same standalone registration entry point automatically detects grayscale and RGB inputs and applies the corresponding matcher.

## Phase 2 Dataset

The Phase 2 datasets are organized into the following sets:

| SetDescription |                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Set A**      | 70 nominal grayscale reference-present pairs with unknown scale (8×–12×) and rotation (±5°).                                     |
| **Set B**      | 70 degraded grayscale reference-present pairs with charging, scan distortion, defocus, elevated shot noise, and polygon scaling. |
| **Set C**      | 40 reference-absent grayscale pairs containing structurally/periodically similar regions to test rejection.                      |
| **Set D**      | 20 **native RGB optical** reference-present pairs representing the official RGB bonus set.                                       |
| **Set E**      | Additional local **pseudo-RGB** extension using grayscale structures with channel variations, blur, and noise.                   |


## Implementation and Execution Commands
**Before running the commands, navigate to the Phase 2 directory:**

```bash
cd PHASE_2
```

### 1. Environment Setup

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate the Synthetic Dataset

Generate the Phase 2 synthetic evaluation dataset:

```bash
python generate_dataset.py --architecture <architecture_name> --output_dir <your_output_dataset_folder> --set-wise
```

For example:

```bash
python generate_dataset.py --architecture mixed --output_dir ./Eval_Dataset --set-wise
```

Available architecture options:

- `dram`
- `finfet`
- `mixed`

If `--architecture` is omitted, the default mixed architecture is used.

The `--set-wise` option generates the required evaluation sets:

- **Set A** — Nominal grayscale reference-present pairs
- **Set B** — Degraded grayscale reference-present pairs
- **Set C** — Reference-absent grayscale pairs
- **Set D** — Native RGB optical reference-present pairs
- **Set E** — Pseudo-RGB extension

### 3. Run the Complete Registration Pipeline

Run the registration pipeline using the image-pair CSV:

```bash
python register.py --input <path_to_pairs.csv> --output <your_output_predictions_file.csv>
```

Each image pair is registered and its predicted pose, scale, confidence score, and found/reject decision are saved in the output CSV.

### 4. Evaluate Performance

Evaluate the registration results:

```bash
python evaluate.py --datasets <path_to_dataset_set_folder(s)> --found-threshold <NCC_detection_threshold> --out_dir <your_evaluation_results_folder> --save-predictions
```

The evaluation produces:

- Localization error
- Sub-pixel localization accuracy
- Scale accuracy
- Rotation accuracy
- Reference found/reject performance
- Confusion matrix
- Runtime statistics
- Prediction files

### Complete Execution Flow

```text
Environment Setup
       ↓
Generate Synthetic Dataset
       ↓
Run Registration Pipeline
       ↓
Generate Predictions
       ↓
Evaluate Performance
       ↓
Analyze Results
```

> **Note:** Replace values enclosed in `< >` with the appropriate architecture, file paths, output folders, or NCC detection threshold for your execution environment.

### Frontend Execution

First, navigate to the frontend directory:

```bash
cd PHASE_2
cd frontend
uvicorn api:app --reload --port 8000
```

## Reference Presence / Rejection

Phase 2 introduces an explicit reference-presence decision.

The registration score is compared against a calibrated threshold:

```
score >= threshold  → found = 1
score < threshold   → found = 0
```

When no reliable match is found, the pose outputs are zeroed according to the Phase 2 output contract.

## Runtime Budget

The Phase 2 addendum requires:

- **Median ≤ 5 s/pair** 
- **Hard timeout = 20 s/pair** 

Our 200-pair evaluation achieved:

| Runtime              |   MetricResult  | 
| -------------------- | --------------- |
| Mean runtime         | **3.62 s/pair** |
| Median runtime       | **3.37 s/pair** |
| Maximum runtime      | **6.38 s/pair** |
| 5 s median target    | **PASS**        |

## Phase 2 Evaluation Results

The combined evaluation contains:

- **200 total pairs** 
- **160 reference-present pairs** 
- **40 reference-absent pairs** 

### Grayscale — Sets A/B/C

- **180 total pairs** 
- **57.14% within 1 px** 
- **62.86% within 5 px** 
-  Median localization error: **0.71 px** 
-  Mean localization credit: **0.6129** 
-  Mean scale credit: **0.5657** 
-  Mean rotation credit: **0.4457** 

### Native RGB — Set D

- **20 native RGB optical pairs** 
- **20.00% within 1 px** 
- **20.00% within 5 px** 
-  Median localization error: **104.02 px** 
-  Mean localization credit: **0.2000** 
-  Mean scale credit: **0.1450** 
-  Mean rotation credit: **0.1650** 

### Reference Rejection

| Metric          | Result     |
| --------------- | ---------- |
| True Positives  | 156        |
| False Positives | 27         |
| True Negatives  | 13         |
| False Negatives | 4          |
| Precision       | **0.8525** |
| Recall          | **0.9750** |
| F1 Score        | **0.9096** |

## Failure Analysis

The primary remaining failure mode is **candidate selection in highly repetitive or periodic semiconductor structures**.

When multiple regions have very similar local appearance, normalized cross-correlation can produce strong responses at incorrect repeated instances. In such cases, the matcher may select a visually similar but incorrect region even though the local refinement is accurate for the selected candidate.

This limitation is particularly important for semiconductor layouts where repeated structures can be nearly indistinguishable without distinctive contextual features.

## Technology and References

`generate_dataset.py` writes a `citations.json` file containing the literature context associated with the generated synthetic pattern families. These references provide technology and layout context for the generation strategy rather than serving as sources of literal numeric generation parameters.

## Phase_2 Files Structure

```
DRIFT-SENSE-REPO/
│
├── documentation/
│   ├── Citation Documents _ Supporting References.pdf
│   └── unique images.pdf
│
├── Eval_Dataset/
│   ├── Set_A/
│   ├── Set_B/
│   ├── Set_C/
│   └── Set_D/
│
├── eval/
│   ├── evaluate.py
│   └── evaluate_predictions.py
│
├── training/
│   ├── precompute_hard_negatives.py
│   └── train_reranker.py
│
├── results/
│   ├── evaluation_details.csv
│   ├── predictions.csv
│   └── failure_cases/
│
├── register.py              
├── matching.py              
├── generate_dataset.py      
├── requirements.txt
├── failure_analysis.md
└── README.md
```
---

# PHASE 1 — Original Submission

## **Demo Video**

[Phase 1 Frontend Demo — Loom](https://www.loom.com/share/250bc63399fd41418f3679019de84057)

## **Tested on 30 Samples**

[Phase 1 — 30 Sample Test Video](https://drive.google.com/file/d/1OfFrgBS9DLr80Lq26Uc-kUlDCYQcnnfX/view)

## **Unique Images**

[svgUnique Images — Google Docs](https://docs.google.com/document/d/19BzJttnVhZkfE4RlggFAp4VYJQfLYAneaWQp9SvFHdA/edit?usp=sharing)

## Overview

Semiconductor inspection systems must repeatedly return to the same wafer location, but small navigation errors (thermal drift, vibration, mechanical inaccuracy) can shift the actual inspection site. Because DRAM, FinFET, and interconnect layouts are highly repetitive, a wrong location can look almost identical to the correct one.

**Task:** given a small high-magnification **Reference Image** and a larger, lower-magnification **Search Image** (\~10× scale difference) of the same layout, find where the reference occurs in the search image and return the center `(x, y)`.

---

## Approach

**Pipeline:** `reference.png + search.png → NCC matching across 45 scale/rotation combos → top-6 peaks per map, non-max suppressed → global merge + rank by score → sub-pixel refine best candidate → (x, y)`

`matching.py` sweeps 9 scales × 5 rotation angles (45 combinations), running NCC template matching at each. From each correlation map it keeps up to 6 local peaks, suppresses near-duplicates, then merges and ranks all peaks globally. The top candidate is refined to sub-pixel accuracy via parabolic interpolation.

**Decision Rule:** pick the top-scoring candidate from the sweep as the final prediction. An optional CNN-reranker mode can instead pick by smallest embedding distance to the reference, breaking ties within `0.02` by choosing the candidate closest to the search image's center — but plain CV top-1 matching performs better and is the shipped default.

---

## Dataset

Synthetic reference/search pairs generated to reproduce realistic wafer-navigation challenges: highly repetitive structures, multiple pattern families (DRAM, FinFET, interconnect), duplicate "twin" regions, and SEM-style degradation (beam blur, drift, shot noise, charging streaks, edge brightening).

```
python generate_dataset.py --architecture dram --num_pairs 30 --output_dir ./synthetic_sem_dataset
```

(Omit `--architecture` for a mixed-pattern dataset by default.)

Each sample includes `reference.png`, `search.png`, `visualization.png`, and `metadata.json`, with ground-truth coordinates in `ground_truth.csv`.

---

## Setup

```
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Environment Setup

```
pip install -r requirements.txt
```

### 2. Generate the Synthetic Dataset

```
python generate_dataset.py --architecture dram --num_pairs 30 --output_dir ./synthetic_sem_dataset
```

(Omit `--architecture` for a mixed-pattern dataset by default.)

### 3. Precompute Hard Negatives (for reranker training)

```
python training/precompute_hard_negatives.py --dataset ./synthetic_sem_dataset --out ./hard_negatives.csv
```

### 4. Train the Reranker Model

This step is optional. The results shown use CV-only matching, so this step can be skipped.

```
python training/train_reranker.py --dataset ./synthetic_sem_dataset --hard_negatives ./hard_negatives.csv --epochs 30 --out ./model/reranker.pt
```

### 5. Run Localization on a Single Pair

```
python localize.py --reference <reference_image_path> --search <search_image_path>
```

Add `--use-cnn --reranker ./model/reranker.pt` to enable the learned reranker. The default mode is pure CV and does not require Torch.

### 6. Evaluate the Dataset

Using the trained CNN reranker:

```
python evaluate.py --dataset ./synthetic_sem_dataset --reranker ./model/reranker.pt --split test --out_csv ./results/evaluation_manifest.csv
```

Without using CNN:

```
python evaluate.py --dataset ./Eval_Dataset --no-cnn --split all
```

Use `--no-cnn` to benchmark the raw CV top-1 candidate without the reranker.

### 7. Launch the Web Demo

```
cd frontend
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000` to upload a reference/search pair and view the localization result.

---

## Results — 30-Test-Case Evaluation

**Hardware:** Intel CPU | Python 3.12 | OpenCV 4.11

| **MetricValue** |                                           |
| --------------- | ----------------------------------------- |
| Within 5 px     | 80.0 %                                    |
| Within 4 px     | 73.3 %                                    |
| Within 2 px     | 63.3 %                                    |
| Within 1 px     | 26.7 %                                    |
| Median error    | 1.80 px                                   |
| Mean error      | 36.04 px (skewed by a few large failures) |
| Runtime         | 0.908 s / 1000×1000 image pair on CPU     |

**Success case —** **`sample_0203`****:** ground truth (182.20, 309.20), prediction (181.95, 309.16), error 0.25 px. Demonstrates accurate sub-pixel localization when the target has enough distinguishing structure.

**Honest failure —** **`sample_1181`****:** ground truth (77.40, 661.90), prediction (740.32, 410.50), error 708.99 px. Caused by a highly repetitive/ambiguous region where the matcher locked onto a visually similar but incorrect location — the fundamental limit of periodic semiconductor layouts when no unique context is present.

**Key takeaway:** the method achieves high-precision, sub-pixel localization on distinguishable regions (1.80 px median error), while highly repetitive regions remain the principal failure mode.

---

## Limitations

- **Synthetic-to-real gap:** the dataset is synthetic; real SEM imagery may include effects not fully captured by the generator. 
- **Fundamental periodic ambiguity:** perfectly periodic regions with no unique context can be genuinely indistinguishable from the reference. 
- **Candidate-generation dependency:** the reranker can only choose among candidates already proposed by the CV stage — if the correct location isn't proposed, it can't be recovered. 

---

## Project Structure

```
DRIFT-SENSE-REPO/
│
├── documentation/
│   ├── Citation Documents _ Supporting References.pdf
│   └── unique images.pdf
│
├── Eval_Dataset/
│
├── frontend/
│   └── api.py
│       └── static/
│           └── index.html
│
├── model/
│   └── reranker.pt
│
├── results/
│   ├── evaluation_manifest_confusion_matrix.csv
│   ├── evaluation_manifest_confusion_matrix.png
│   ├── evaluation_manifest.csv
│   └── failure_case_sample_1181.png
│
├── training/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── precompute_hard_negatives.py
│   └── train_reranker.py
│
├── evaluate.py
├── generate_dataset.py
├── localize.py
├── matching.py
├── README.md
└── requirements.txt
```
