# SEM Navigation Error Recovery — Wafer Site Localization

**Demo Video:** `[VIDEO_LINK_HERE]`
**Dataset:** `[DATASET_LINK_HERE]`
**Inference / Demo:** `[INFERENCE_LINK_HERE]`

## Overview

Modern semiconductor inspection systems need to repeatedly return to exactly the same location on a wafer. Small navigation errors caused by thermal expansion, vibration, environmental effects, or mechanical inaccuracies can move the inspection location away from the intended site.

The problem becomes particularly difficult for semiconductor layouts because DRAM, FinFET, interconnect, and standard-cell structures are highly repetitive. A wrong location can therefore look almost identical to the correct location.

Our task is to take a **Reference Image** and a larger **Search Image**, locate where the reference occurs inside the search image, and return the center `(x, y)` of that location in Search Image pixel coordinates.

The reference and search images represent the same physical layout at approximately a **10× scale difference**, making this a scale-aware localization problem rather than a simple template-matching problem.

When multiple candidates are visually indistinguishable or nearly equally good, the problem specification requires selecting the candidate closest to the center of the Search Image.

---

## Our Approach

We developed the solution in stages.

We first explored a classical computer-vision based matching approach. This provided useful candidate locations, but the highly periodic nature of semiconductor layouts resulted in false matches: a visually strong match was not necessarily the correct physical site.

We then introduced a learned CNN-based reranking stage. During development, we found an important issue with directly comparing the high-resolution reference with candidate search patches. Because the two images represent very different physical scales, resizing the full reference directly to the CNN input size caused the reference to lose substantially more spatial detail than the candidate patch.

Our current solution combines the strengths of both approaches:

**Classical CV candidate generation → scale-aware CNN reranking → centre-based tie breaking**

The overall pipeline is:

```text
Reference Image + Search Image
              │
              ▼
      Candidate Generation
              │
              ▼
       Top-K Candidates
              │
              ▼
   Scale-aware Reference
       Preprocessing
              │
              ▼
       CNN Embeddings
              │
              ▼
     Embedding Distance
              │
              ▼
      Candidate Ranking
              │
              ▼
     Centre-based Tie Break
              │
              ▼
        Predicted (x, y)
```

---

# Dataset

## Why We Created a Synthetic Dataset

Our dataset is synthetically generated specifically for the SEM navigation/localization problem.

Instead of using a collection of unrelated images, we generate an underlying semiconductor-style layout and then create corresponding Reference and Search images from that same physical world.

This allows us to control:

* the physical scale difference
* pattern geometry
* layout repetition
* difficulty
* landmarks and contextual information
* image degradation
* search-image drift
* ambiguity caused by repeated structures

This is important because the main challenge is not simply finding a visually similar image. The algorithm must determine **which occurrence of a repeated structure corresponds to the intended physical location**.

---

## Reference and Search Scale

Our current generator uses:

```python
REF_PX = 1000
SEARCH_PX = 1000

REF_NM_PER_PX = 1.0
SEARCH_NM_PER_PX = 10.0
```

Therefore, both images are stored at `1000 × 1000` pixels, but they represent very different physical fields of view.

The Reference Image has a spatial resolution of approximately **1 nm/pixel**, while the Search Image represents approximately **10 nm/pixel**.

The resulting scale ratio is:

```text
10 / 1 = 10×
```

The Search Image therefore covers a much larger physical region while using the same number of pixels.

This scale difference is one of the central difficulties addressed by our final model.

---

# Generating Our Dataset

The current generator produces **500 reference/search pairs by default**.

To change the number of samples, modify:

```python
N_SAMPLES = 500
```

at **line 139** of the dataset-generation code.

For example:

```python
N_SAMPLES = 1000
```

will generate 1000 samples.

The default output directory is controlled by:

```python
OUT_ROOT = "./synthetic_sem_dataset_v6"
```

at **line 142**.

To save the generated dataset somewhere else, change this value:

```python
OUT_ROOT = "./my_dataset"
```

The generator also exposes command-line options for controlling the generation process:

```text
--style
--num_pairs
--output_dir
--seed
```

For example:

```bash
python dataset_generator.py --num_pairs 500 --output_dir ./synthetic_sem_dataset_v6 --style mixed
```

The exact entry-point filename should be replaced with the dataset-generation entry point in the repository if it has been renamed.

### Architecture styles

The generator supports three architecture-style modes:

```text
dram
finfet
mixed
```

### DRAM

The DRAM mode generates patterns from the DRAM-oriented family, including repeated storage-cell structures, staggered structures, wavy bitlines, and layered word-line/bit-line structures.

### FinFET

The FinFET mode focuses on fin/gate structures and logic-oriented patterns such as standard-cell-style tracks.

### Mixed

The default `mixed` mode provides the largest diversity by combining DRAM, FinFET, interconnect, contact, and logic-oriented pattern families.

---

# What Makes Our Dataset Challenging?

Our dataset is designed to reproduce the main sources of difficulty in the navigation problem.

## 1. Highly Repetitive Semiconductor Structures

The generator creates repeated structures rather than unique natural-image-like objects.

Examples include:

* DRAM arrays
* FinFET-like fin and gate structures
* contact arrays
* ring arrays
* interconnect meshes
* layered cells
* logic stripes
* standard-cell-like structures
* BEOL interconnect patterns

This creates realistic ambiguity: several regions can have extremely similar visual appearance.

---

## 2. Multiple Pattern Families

The dataset does not rely on one fixed synthetic pattern.

The generator includes several pattern families with randomized parameters such as pitch, linewidth, feature dimensions, contact sizes, array spacing, and other geometric properties.

This prevents the task from becoming a simple lookup problem based on one fixed pattern.

---

## 3. Repeated / Twin Structures

At higher difficulty levels, the generator can intentionally duplicate patterns in multiple regions of the same underlying world.

This creates exactly the type of failure that is important for the problem:

```text
Reference
   ↓
Candidate A ── visually similar
Candidate B ── visually similar
Candidate C ── visually similar
   ↓
Which one is the correct physical location?
```

A conventional similarity score alone may not be sufficient to resolve this ambiguity.

---

## 4. Context and Landmarks

The generator can introduce landmark-like structures into the layout.

It can also create reference crops that straddle internal streets or boundaries.

These cases provide additional contextual information that can help distinguish otherwise repetitive regions.

The evaluation therefore separately considers cases with and without this type of disambiguating context.

---

## 5. Difficulty Progression

The generator uses five difficulty levels:

| Level | Description  |
| ----- | ------------ |
| 1     | Easy         |
| 2     | Moderate     |
| 3     | Hard         |
| 4     | Very Hard    |
| 5     | Failure Case |

Difficulty affects several aspects of the generated problem, including the amount of image degradation, drift, ambiguity, and other challenging conditions.

---

# SEM-like Image Degradation

The synthetic images are not simply clean binary layouts.

The generator applies several image-degradation effects intended to reproduce useful characteristics of SEM imagery.

Depending on the generated sample, these include:

* beam blur
* astigmatism
* dose/shot noise
* raster drift
* row jitter
* shear
* geometric distortion
* charging streaks
* speckle noise
* salt-and-pepper noise
* edge brightening

Reference and Search images receive independent noise/profile draws so that the model cannot rely on identical noise patterns.

### Edge Brightening

One specific addition in our current dataset generator is SEM-style edge brightening.

We calculate the image gradient using Sobel filters, smooth the gradient magnitude, normalize it, and add the resulting edge mask back to the image.

Conceptually:

```text
Image
  ↓
Sobel X + Sobel Y
  ↓
Gradient Magnitude
  ↓
Smooth + Normalize
  ↓
Add to Image
  ↓
Brighter Feature Edges
```

This is intended as a synthetic approximation of brighter contrast around feature boundaries.

It is not intended to be a physically exact SEM simulator.

---

# Dataset Structure

Each generated sample contains the reference image, search image, visualization, and metadata.

The generated dataset follows the general structure:

```text
synthetic_sem_dataset_v6/
├── sample_0000/
│   ├── reference.png
│   ├── search.png
│   ├── visualization.png
│   └── metadata.json
│
├── sample_0001/
│   ├── reference.png
│   ├── search.png
│   ├── visualization.png
│   └── metadata.json
│
├── ...
│
├── ground_truth.csv
└── citations.json
```

### `reference.png`

The high-resolution Reference Image that the localization system needs to find inside the Search Image.

### `search.png`

The larger-field-of-view Search Image containing the target region and potentially many visually similar regions.

### `visualization.png`

A visualization showing the generated ground-truth target location.

### `metadata.json`

Contains information about the generated world, difficulty, generation mode, layout, noise, context, and ground-truth coordinates.

### `ground_truth.csv`

Contains the information required for training and evaluation, including the target coordinates.

The target center is represented using:

```text
GT_X
GT_Y
```

in Search Image pixel coordinates.

The coordinate system is:

```text
(0,0) ───────────────► x
  │
  │
  │
  ▼
  y
```

---

# Preventing Data Leakage

A major consideration in our dataset is that multiple samples can potentially originate from the same underlying generated world.

To prevent this from making the evaluation artificially easy, the generator stores a `world_id` identifying the underlying pattern world.

The intended train/validation/test split is therefore based on the underlying world rather than simply treating every crop as an independent sample.

This means that the held-out test set contains samples originating from worlds that were not used during training or model selection.

This gives us a more meaningful measure of generalization.

---

# What We Tried First

## Classical Matching Baseline

Our initial approach relied primarily on classical computer-vision matching.

The idea was straightforward:

```text
Reference
    ↓
Multi-scale / CV matching
    ↓
Candidate locations
    ↓
Best matching location
```

This approach was useful because it could efficiently identify visually similar regions.

However, the semiconductor layouts exposed an important weakness.

A repeated DRAM or FinFET structure can produce several equally strong matches:

```text
       Search Image

   ┌─────┐  ┌─────┐  ┌─────┐
   │     │  │     │  │     │
   │  A  │  │  B  │  │  C  │
   │     │  │     │  │     │
   └─────┘  └─────┘  └─────┘
                 ↑
              Target
```

The strongest image similarity does not necessarily identify the correct physical site.

We therefore observed high errors in difficult periodic cases, particularly when the reference contained little unique contextual information.

---

# Why We Moved to a Learned Reranker

Rather than asking a CNN to search the entire Search Image from scratch, we use the strengths of both approaches.

Classical CV is used for **candidate generation**.

The learned model is then used to answer a more specific question:

> Among these promising candidates, which one looks most similar to the Reference Image?

This reduces the search space while giving the learned model a chance to distinguish candidates that are visually similar but not identical.

---

# Our Current Model

Our final pipeline is:

```text
Reference + Search
        │
        ▼
Classical CV Candidate Generation
        │
        ▼
Top 30 Candidates
        │
        ▼
Scale-aware CNN Comparison
        │
        ▼
Embedding Distance
        │
        ▼
Candidate Reranking
        │
        ▼
Centre-based Tie Breaking
        │
        ▼
Final (x, y)
```

## Stage 1 — Candidate Generation

The current system first generates up to **30 candidate locations**.

This prevents the CNN from having to evaluate every possible position in the Search Image.

The candidate generator also estimates information such as candidate position and scale.

---

## Stage 2 — Scale-aware Reference Processing

This was one of the most important changes from our earlier approach.

The Reference Image is approximately 10× higher resolution than the Search representation.

If we directly resize the complete reference to the CNN input size, a large amount of spatial detail is lost.

At the same time, the candidate patch extracted from the Search Image has already undergone a different amount of scale reduction.

Therefore, the CNN may receive two inputs containing substantially different levels of detail.

### Our solution

For every candidate, we first downsample the Reference Image according to that candidate's estimated scale.

```text
Full-resolution Reference
          │
          ▼
Candidate-specific Downsampling
          │
          ▼
Resize to CNN Input
          │
          ▼
Reference Embedding
```

The candidate patch is processed separately:

```text
Search Image
     │
     ▼
Candidate Region
     │
     ▼
Resize to CNN Input
     │
     ▼
Candidate Embedding
```

Now the reference and candidate undergo a much more comparable amount of information loss before being compared.

The current implementation also caches reference embeddings for repeated candidate scales, avoiding unnecessary recomputation.

---

# CNN Embedding Model

Our current method uses a lightweight CNN embedding model.

The model maps an image patch to a compact feature representation.

The final evaluation pipeline uses a:

```text
128-dimensional embedding
```

For every candidate, we obtain:

```text
Reference embedding
        +
Candidate embedding
        ↓
Embedding distance
```

The goal is for visually corresponding regions to have smaller embedding distances.

The exact architecture and training configuration are defined by our current training implementation.

---

# Candidate Reranking

For every generated candidate, our current model:

1. Determines the candidate scale.
2. Prepares the reference at that scale.
3. Extracts the corresponding candidate patch.
4. Converts both into CNN inputs.
5. Generates embeddings.
6. Computes their squared L2 distance.
7. Ranks candidates according to that distance.

The distance used by the current implementation is:

```text
D = Σ (ReferenceEmbeddingᵢ - CandidateEmbeddingᵢ)²
```

A smaller value means that the two embeddings are more similar.

---

# Centre-based Tie Breaking

The problem specification contains an important rule for repeated structures.

Sometimes multiple candidates are effectively equally good matches.

Our current model handles this using:

```python
CENTER_TIE_MARGIN = 0.02
```

Candidates whose embedding distance is within `0.02` of the best candidate are considered tied.

Among these candidates, we select the candidate closest to the center of the Search Image.

Therefore:

```text
Best embedding match
        ↓
Are there near-equivalent candidates?
        │
       Yes
        ↓
Choose the one closest to Search Image centre
        ↓
Final prediction
```

This explicitly incorporates the problem's requirement instead of relying solely on visual similarity.

---

# Training

Our CNN reranker is trained using examples derived from our generated reference/search data.

The training process teaches the embedding model to produce representations that allow the correct reference/candidate pair to be distinguished from incorrect candidates.

The training pipeline produces a trained model checkpoint which is then used during localization.

The exact training command and hyperparameters should be taken from the current training configuration in the repository.

---

# Running the Current Model

The final evaluation/localization pipeline can be run using:

```bash
python evaluate.py --dataset ./synthetic_sem_dataset --reranker reranker.pt
```

The default behavior evaluates the held-out test split.

To evaluate the entire dataset for a sanity check:

```bash
python evaluate.py --dataset ./synthetic_sem_dataset --split all
```

The whole-dataset result should **not** be used as the main generalization result because it includes worlds used during training/model selection.

---

# CV-only Ablation

We also support evaluating the system without the CNN reranker:

```bash
python evaluate.py --dataset ./synthetic_sem_dataset --no-cnn
```

This uses the raw CV candidate ranking.

This comparison is important because it tells us whether the learned reranking stage is actually helping.

Our intended comparison is:

```text
CV-only
   vs.
CV + CNN reranking
```

rather than assuming that adding a neural network automatically improves performance.

---

# Evaluation

Our primary evaluation uses the **held-out test split** containing unseen underlying worlds.

For each sample, we compare:

```text
Predicted (x, y)
        vs.
Ground-truth (x, y)
```

The localization error is the Euclidean pixel distance:

```text
Error = √[(x_pred - x_GT)² + (y_pred - y_GT)²]
```

We report:

* Mean pixel error
* Median pixel error
* Percentage within 5 pixels
* Percentage within 4 pixels
* Percentage within 2 pixels
* Percentage within 1 pixel
* Average runtime per sample

Performance is also broken down by:

* difficulty level
* generation mode
* availability of disambiguating context

---

# Why Context Matters

One of the most important observations from this problem is that not every localization error is simply a model failure.

Consider a perfectly periodic layout:

```text
| A | A | A | A | A |
```

If the Reference Image contains only one repeated `A` and no unique surrounding context, multiple positions may be visually identical.

In such a case, there is no image-only information that can reliably tell the algorithm which occurrence was the intended one.

This is why we report performance separately for:

**Cases with disambiguating context**

and

**Pure periodic / no-context cases**

rather than hiding this distinction inside a single overall accuracy number.

---

# Results

## Final Held-out Test Results

| Metric                   | Our Current Model |
| ------------------------ | ----------------: |
| Test Samples             |        `[INSERT]` |
| Mean Pixel Error         |        `[INSERT]` |
| Median Pixel Error       |        `[INSERT]` |
| Within 5 px              |        `[INSERT]` |
| Within 4 px              |        `[INSERT]` |
| Within 2 px              |        `[INSERT]` |
| Within 1 px              |        `[INSERT]` |
| Average Runtime / Sample |        `[INSERT]` |

## Comparison

| Metric           | Previous Approach |    CV-only | Current Model |
| ---------------- | ----------------: | ---------: | ------------: |
| Mean Pixel Error |        `[INSERT]` | `[INSERT]` |    `[INSERT]` |
| Within 5 px      |        `[INSERT]` | `[INSERT]` |    `[INSERT]` |
| Within 2 px      |        `[INSERT]` | `[INSERT]` |    `[INSERT]` |
| Within 1 px      |        `[INSERT]` | `[INSERT]` |    `[INSERT]` |

These values should be filled using the actual experimental results rather than estimated values.

---

# Accuracy by Difficulty

Our evaluation also reports performance across the five difficulty levels.

| Difficulty   |    Samples | Mean Error | Within 5 px |
| ------------ | ---------: | ---------: | ----------: |
| Easy         | `[INSERT]` | `[INSERT]` |  `[INSERT]` |
| Moderate     | `[INSERT]` | `[INSERT]` |  `[INSERT]` |
| Hard         | `[INSERT]` | `[INSERT]` |  `[INSERT]` |
| Very Hard    | `[INSERT]` | `[INSERT]` |  `[INSERT]` |
| Failure Case | `[INSERT]` | `[INSERT]` |  `[INSERT]` |

---

# Failure Case

For every evaluation run, we identify the sample with the largest localization error.

The failure visualization marks:

* **Green:** Ground-truth location
* **Red:** Model prediction

**Failure case:**

`[INSERT_FAILURE_CASE_IMAGE_HERE]`

The failure case is useful for understanding where the current approach still struggles, particularly when repeated structures provide insufficient contextual information or when the correct candidate is not successfully proposed during the initial candidate-generation stage.

---

# Reproducibility

A complete reproduction of our approach follows these steps:

```text
1. Clone the repository
        ↓
2. Install the required dependencies
        ↓
3. Generate our synthetic dataset
        ↓
4. Train the CNN reranker
        ↓
5. Run the final localization/evaluation
        ↓
6. Inspect predictions, metrics and failure cases
```

### Step 1 — Generate the dataset

Set the desired number of samples using:

```python
N_SAMPLES = 500
```

at line 139.

Set the output directory using:

```python
OUT_ROOT = "./synthetic_sem_dataset_v6"
```

at line 142.

Or use the supported command-line arguments for sample count, output directory, architecture style, and seed.

### Step 2 — Train

Run the training procedure provided in the repository to generate the reranker checkpoint.

### Step 3 — Evaluate

```bash
python evaluate.py --dataset ./synthetic_sem_dataset --reranker reranker.pt
```

The default evaluation uses the held-out test split.

---

# Project Structure

At a high level, our repository is organized around the following components:

```text
Project
│
├── Dataset Generation
│   └── Synthetic SEM-like reference/search pairs
│
├── Candidate Generation
│   └── Classical computer-vision matching
│
├── Model Training
│   └── CNN embedding / reranker
│
├── Localization
│   └── Candidate reranking + centre tie-break
│
├── Evaluation
│   └── Pixel error, pass rates and failure analysis
│
└── README
```

The important idea is that the system is modular:

```text
Dataset
   ↓
Candidate Generation
   ↓
Learned Reranking
   ↓
Localization
   ↓
Evaluation
```

---

# Method at a Glance

```text
Synthetic Semiconductor Layout
              │
              ▼
      Reference + Search
              │
              ▼
      10× Scale Difference
              │
              ▼
    Classical CV Candidates
              │
              ▼
       Top 30 Candidates
              │
              ▼
 Candidate-specific Reference
       Downsampling
              │
              ▼
       CNN Embeddings
              │
              ▼
     Squared L2 Distance
              │
              ▼
      Candidate Reranking
              │
              ▼
      0.02 Tie Margin
              │
              ▼
  Closest Candidate to Centre
              │
              ▼
       Final (x, y)
```

---

# Limitations

Our current approach has several important limitations.

### Synthetic-to-real gap

Our dataset is synthetic. Although we intentionally model semiconductor structures and SEM-like degradation, real SEM imagery can contain physical effects that are not fully represented by our generator.

### Fundamental periodic ambiguity

If a region is perfectly periodic and contains no unique contextual information, multiple locations can genuinely be indistinguishable from the Reference Image.

### Candidate-generation dependency

Our CNN reranker only evaluates the candidates produced by the initial candidate-generation stage.

Therefore, if the correct location is not present among the generated candidates, the CNN cannot recover it.

### Computational cost

The current evaluation pipeline performs candidate generation followed by CNN scoring and is currently evaluated on CPU.

---

# Future Work

Possible improvements to the current system include:

* validation on real SEM datasets
* stronger metric-learning architectures
* improved multi-scale feature extraction
* learned candidate generation
* transformer-based feature matching
* GPU-accelerated inference
* uncertainty estimation for predicted coordinates
* more physically detailed SEM image simulation
* improved handling of fundamentally periodic regions

---

# Citations and Technical Background

Our synthetic pattern families are motivated by semiconductor technology and layout literature, including resources related to:

* DRAM structures
* FinFET device geometry
* advanced-node scaling
* standard-cell structures
* contact/via arrays
* BEOL interconnects
* semiconductor technology roadmaps

These references provide **technology and layout context** for our synthetic generation strategy. They should not be interpreted as claiming that the exact numerical values used for our synthetic pitches, noise parameters, or other generator parameters were directly taken from those sources.

---

# Links

**Demo Video:** `[VIDEO_LINK_HERE]`

**Dataset:** `[DATASET_LINK_HERE]`

**Inference / Demo:** `[INFERENCE_LINK_HERE]`

---

# Acknowledgement

This project was developed as a solution to the SEM navigation error recovery / wafer site localization problem, with the goal of combining classical computer vision and learned visual representations to handle the scale difference and severe periodic ambiguity present in semiconductor layouts.
