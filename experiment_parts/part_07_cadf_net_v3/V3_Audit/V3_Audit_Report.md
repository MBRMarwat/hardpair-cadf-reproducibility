# CADF-Net v3 focused post-hoc audit

## Scope and safeguards

This audit used only the completed CADF-Net v3 checkpoints and saved outputs. No model was retrained, no parameter or threshold was tuned, and no original result was overwritten. The held-out HAM10000 test labels were used only for diagnosis and reporting.

## Main finding: the apparent Stage 3 collapse mixed checkpoint states

The saved training-history values of 41.94% (Stage 2) and 26.97% (Stage 3) describe the final epochs. They are not the checkpoints selected and restored by the pipeline. Stage 2 selected epoch 4; Stage 3 selected epoch 2.

A fresh deterministic image pass gave 37.93% for the Stage 2 last checkpoint and 26.97% for Stage 3 last. Reapplying the Stage 2 heads to the original saved feature cache also gives 37.93%, so the discrepancy is not caused by reconstructing image features.

Code inspection identifies the cause: the Stage 2 loop calls `specialist_head.train()`, then `_head_validation()` evaluates under `torch.no_grad()` without calling `model.eval()`. The generalist and specialist head dropout layers therefore remain active during the saved Stage 2 validation history and checkpoint selection. The 41.94% value is a stochastic dropout-on measurement; 37.93% is the deterministic dropout-off value from the saved final checkpoint. Stage 3 calls `model.eval()` and its 26.97% final-epoch value reproduces. This is a validation/checkpoint-selection defect, not evidence about feature-cache drift.

Under one identical image-based definition, the selected Stage 2 checkpoint corrected 43 of 87 conditional MEL/NV errors (49.43%). The selected Stage 3 checkpoint corrected 44 of 99 (44.44%). Because joint fine-tuning changes the generalist as well as the specialist, these raw denominators differ; the fixed-denominator tables must be used for a direct specialist comparison.

On the fixed set of 87 images that were Stage 2 conditional errors, Stage 2's specialist was correct on 43 and Stage 3's specialist on 36: a net loss of 7 corrections (-8.05 percentage points). On the 73 errors shared by both generalists, the change was 35 to 26 (-12.33 points). In contrast, specialist accuracy across all 803 MEL/NV validation images rose from 76.71% to 87.80%. Joint fine-tuning therefore improved broad pair classification while weakening performance specifically on the difficult cases the generalist got wrong, which supports the hypothesized complementarity conflict.

The final model bundle is exactly identical to the saved Stage 3 best model state (maximum tensor difference 0).

## Gate and feasibility-transfer audit

On 1010 held-out test images, learned alpha fusion changed 10 seven-class predictions. Of those changes, 3 were beneficial and 4 were harmful. Adding the learned beta transfer changed 2 predictions beyond primary fusion: 0 beneficial, 1 harmful, and 1 wrong-to-wrong.

Although mean beta was 0.4446, the mean realized MEL probability transfer was only 0.000420; 98 images received more than 0.001 MEL probability and 2 received more than 0.01. This demonstrates why a large beta coefficient alone is not evidence of a material transfer: beta is multiplied by the off-pair margin and three soft gates.

## Validation-to-test gap

Full CADF-Net v3 pair-balanced accuracy changed from 66.01% on validation to 61.11% on test (difference -4.89 percentage points). MEL sensitivity changed from 46.61% to 36.00%.

The lesion-level 95% bootstrap interval for the full-model pair-balanced-accuracy gap is [-12.68, 3.49] percentage points; for the MEL-sensitivity gap it is [-25.87, 5.46]. Both include zero. These intervals quantify sampling uncertainty, but this observational comparison cannot by itself separate sampling variation, cohort composition, or representation shift. It should not be interpreted as a causal attribution.

## Interpretation

The audit distinguishes three claims. First, final-epoch training history and selected-checkpoint behavior are different quantities and must not be compared as if they were the same model states. Second, the fixed-denominator paired analysis measures whether the specialist itself gained or lost corrective decisions; the varying-denominator rates additionally include changes in the generalist error set. Third, learned beta magnitude is not the realized transfer magnitude, and the prediction-change table directly measures its practical effect.

These results are diagnostic, not a basis for another round of test-guided tuning. Any architectural change should be selected on development data and assessed on a new independent cohort.

## Reproducibility

Exact source hashes are recorded in `results/source_manifest.csv`. Machine-readable findings are in `results/audit_summary.json`; all checkpoint-level, per-image, fixed-denominator, class-stratified, and bootstrap results are preserved in the other CSV files.
