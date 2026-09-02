# Failure Analysis -- Drift-Sense Phase 2

## Method and rationale

`register.py` uses classical multi-scale/multi-rotation normalized cross-
correlation (`matching.py`) with sub-pixel refinement and a nearest-to-
centre tie-break, extended from our Phase 1 declared approach to search
the full disclosed `[8,12]x` / `+-5deg` pose space. No CNN component is
used in the graded path.

## CV vs. CNN: two rounds of evidence, both favoring classical CV

A Siamese-triplet CNN reranker (`TinyEmbedNet`, 4-conv-block; and a deeper
`ImprovedEmbedNet` with residual blocks + dropout) was trained on our own
generated present-pair data (mined hard negatives, seeded training, LR
scheduling, early stopping, up to 40 epochs) and evaluated against the
same held-out test split as the CV baseline.

| Method | Within 5px | Mean error | Val acc (training) |
|---|---|---|---|
| Classical CV (flat grid + centre tie-break) | **17.8%** | 352px | n/a |
| TinyEmbedNet reranker | 0.0% | 379px | plateaued ~0.55-0.60 |
| ImprovedEmbedNet reranker | 0.0% | 442px | plateaued ~0.55-0.60 |

Validation accuracy for both CNN variants never broke meaningfully past
~0.55-0.60 (barely above the 0.50 chance baseline for "is the positive
embedding closer than the negative"), despite seeding, scheduling, and a
deeper architecture. Paired inspection shows the CNN actively **overriding
correct CV picks with wrong ones**: on samples where CV localized to
<1.5px error with high NCC confidence, the reranker frequently moved the
answer to a different, wrong candidate (e.g. sample `0_sample_0059`: CV
err=0.83px score=0.948 -> tiny reranker err=547.6px score=0.807). This
happened repeatedly enough to conclude the learned embedding space carries
essentially no reliable signal for this domain -- most patches in this
dataset are periodic repeats that are close to visually indistinguishable
from their neighbors at 96x96 resolution, which may leave little
learnable signal beyond position, something the reranker has no access to.

Given this pattern held across two separate phases and two architectures,
we did not pursue transfer learning from ImageNet-pretrained backbones:
the domain gap (synthetic grayscale periodic SEM patterns vs. natural
images) makes positive transfer speculative, and a larger backbone would
also risk the 5s/pair runtime budget for a payoff we have no evidence for.

## Selection-rule bug found and fixed mid-development

An earlier version of our CV pipeline used pure argmax (highest NCC score
wins) with no tie-break. The addendum states the same nearest-to-centre
rule from Phase 1 still applies, and that Set B "deliberately includes
pairs where the global-argmax answer differs from the nearest-to-centre
answer." Adding the tie-break (pull toward image centre among candidates
within 0.03 NCC of the top score) recovered a measurable share of large
errors caused by picking the wrong instance of a periodic/twin-duplicate
pattern. An adaptive variant (only tie-break when the top-2 scores are
close, trust a clear standout outright) combined with a coarse-to-fine
search was also tried, purely for speed; it measured **worse** on held-out
eval (14.4% within 5px vs. 17.8% for the flat grid + unconditional
tie-break), so the simpler, slower, more accurate version is what shipped.

## Candidate recall is the dominant remaining gap

Overall within-5px accuracy (17.8%) is well below our Phase 1 number
(78.6%) and below the organizer's own worked-reference baseline (80.0%
mean credit on their 20-pair sample set, though that set was hand-curated
and, per their own README, "too easy" for a 200-pair grading set). The
gap traces mostly to candidate recall, not selection: `matching.py`'s own
in-code measurement on a smoke set found the true match present in the
top-30 candidate pool only ~50% of the time. No selection-rule
improvement can recover a candidate that generate_candidates() never
produced. `single_field`-mode samples (one continuous periodic field, no
distinguishing landmark) are close to 0% recall by construction --
genuinely ambiguous, matching the "several tiles match" scenario the
problem statement itself describes.

## Known limitations / not fully resolved

1. **Runtime budget risk.** Development-machine median runtime was
   ~3.8-5.4s/pair across runs, close to or over the 5s addendum budget
   (well under the 20s hard timeout). The wider Phase 2 search space
   (17 scale steps x 11 rotation steps = 187 correlation-map evaluations
   per pair, vs. 45 in Phase 1) is the direct cause. A coarse-to-fine
   search was tried to address this but reduced accuracy more than it
   helped runtime, so it was not adopted; this remains an open risk on
   the actual reference machine, which may differ in raw throughput from
   our development machine.
2. **Recall on `single_field` and dense-periodic patterns** is the
   dominant source of remaining localization error, as above -- addressed
   only partially (denser peak extraction, tighter suppression radius),
   not solved.
3. **Rejection/found threshold (0.335)** was calibrated on our own
   generated Set A/B/C data (`training/calibrate_rejection.py`, F1=0.916
   on that split), not on the organizer's actual distribution, which may
   differ.
4. **A generator bug was found and fixed during development**: an earlier
   generator version computed the ground-truth bounding box before
   applying search-image rotation and never recomputed it afterward, so
   `GT_X_min/Y_min/X_max/Y_max` were stale (pixel-wrong) whenever
   `theta_deg != 0`, while `GT_X/GT_Y` (centre) and `GT_bbox_corners` were
   correct. This is fixed in the shipped `generate_dataset.py` (box is
   now the axis-aligned envelope of the rotated corners) -- flagged here
   because it's exactly the class of labeling bug the organizer's own
   Phase 2 sample-pairs README calls out as needing an explicit
   verification gate to catch.

## Smoke-test finding: rejection is not reliably threshold-separable

A 3-pair smoke test of Set-C-style absent pairs (style-matched decoys --
same pattern family as the reference, independently seeded) found
`register.py` incorrectly reporting `found=1` on 2 of 3 genuinely absent
pairs, with scores (0.856, 0.793) well above the calibrated 0.335
threshold and comparable to typical present-pair scores. This sample is
far too small to be conclusive on its own, but it is directly consistent
with the organizer's own worked-reference finding: present-pair scores
and absent-pair scores overlap (their reported separation gap was
negative: the two most degraded present pairs scored below their
strongest absent pair). Style-matched decoys -- built to be genuinely
periodically similar to the reference, per the addendum's own wording --
are close to adversarial for a pure top-1-NCC threshold. The 0.335
threshold and 0.916 F1 reported in this repo were calibrated on 750
samples and should be treated as an average-case number, not a guarantee;
expect real false positives on hard decoys, and treat rejection as the
scoring block most likely to underperform its calibration number on the
organizer's actual blind set.

## What we'd do next with more time

Audit and improve candidate recall directly (the ~50% top-30 hit rate)
rather than further tuning selection logic on top of it; profile
`generate_candidates()` for a genuine speedup that doesn't cost accuracy
(vectorizing the scale/rotation loop, or precomputing rotated templates
once per reference rather than per scale); and generate a much larger,
more diverse `single_field` sample set to see whether recall there is
fundamentally near-zero or just under-served by current peak-extraction
parameters.
