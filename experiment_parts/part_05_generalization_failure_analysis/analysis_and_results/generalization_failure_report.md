# HAM10000 to BCN20000 generalization-failure analysis

## Scope and integrity

This is a post-hoc analysis of saved predictions, probabilities, ImageNet-pretrained feature caches, metadata, and image summary statistics. Neither frozen disease classifier was run, retrained, recalibrated, or retuned. The small diagnostic domain classifiers predict dataset identity, not diagnosis.

## Main result

Frozen-baseline image-level MEL sensitivity fell from **40.8%** on the HAM10000 held-out test set (125 MEL images) to **22.1%** on BCN20000 (2857 MEL images), a change of **-18.7 percentage points**. A lesion-level bootstrap gives -18.7 points (95% CI -29.8 to -7.2).

After averaging probabilities within each MEL lesion, sensitivity is 43.9% on HAM and 18.9% on BCN. The failure therefore is not explained solely by BCN having more images per lesion.

## Why melanomas become BKL and NV

The true-MEL-to-BKL rate rises from 15.2% to 30.9% (+15.7 points; 95% CI +7.2 to +23.5). The MEL-to-NV rate changes from 43.2% to 43.4% (+0.2 points; 95% CI -11.8 to +12.2).

Conditional MEL-over-NV preference falls from 52.0% to 37.9%. At the same time, the proportion of MEL images whose total MEL+NV probability mass cannot overtake the strongest off-pair class rises from 12.8% to 28.8%. Thus the external loss contains both weaker MEL/NV discrimination and a larger off-pair allocation problem; BKL errors cannot be repaired by merely choosing between MEL and NV when pair mass is too low.

On BCN, 62.1% of MEL images are wrong with NV already preferred within the pair. Another 13.5% prefer MEL over NV but are mathematically unrecoverable under the pair-mass constraint. A further 2.3% are wrong yet theoretically recoverable by pair-restricted redistribution.

## Dataset characteristics versus representations

- Reweighting BCN MEL images to the HAM distribution of **age_bin** gives 22.0% sensitivity, leaving a -18.8-point residual. This one-factor descriptive standardization does not remove the failure.
- Reweighting BCN MEL images to the HAM distribution of **sex** gives 22.1% sensitivity, leaving a -18.7-point residual. This one-factor descriptive standardization does not remove the failure.
- Reweighting BCN MEL images to the HAM distribution of **site** gives 20.5% sensitivity, leaving a -20.3-point residual. This one-factor descriptive standardization does not remove the failure.

Geometry/resolution alone separates the cohorts perfectly (AUC 1.000) because HAM images are 600x450 while the local BCN images are 1024x1024. After excluding geometry and using only thumbnail photometric summaries, diagnosis-matched domain AUC is still 0.831. ImageNet-pretrained 576-dimensional features yield 1.000, and the frozen baseline probability vector yields 0.725. For melanoma-only matched samples the photometric, pretrained-feature, and frozen-output AUCs are 0.820, 1.000, and 0.678.

These high domain-separability measurements document dataset shift in both simple image characteristics and representation/output spaces. They are **not additive variance components** and cannot support a statement such as “X% is caused by the dataset and Y% by the model.” Dataset acquisition affects representations, and the frozen representation/decision rule determines how that shift becomes classification error.

## Defensible conclusion

The sensitivity loss is not a gate-specific failure. It is already present in the frozen seven-class baseline and reflects two coupled changes: external melanomas receive less MEL/NV pair mass, especially relative to BKL, and the MEL-versus-NV ordering itself is less reliable. Available age, sex, and site composition differences do not explain away the decline under one-factor reweighting. The saved feature and image summaries show substantial cohort separability, supporting domain shift as a major context, while the probability diagnostics show exactly how the learned decision representation converts that shift into BKL and NV errors.

## Limitations

This is observational and post-hoc. Metadata fields are coarsened and not perfectly harmonized. Perceptual-hash overlap candidates are screening hits rather than confirmed duplicate lesions. Image summaries are low-dimensional proxies for acquisition conditions. Fine-tuned internal feature vectors were not saved, so representation-space diagnostics use the saved ImageNet-pretrained features and frozen model outputs rather than hidden fine-tuned activations. Domain-classifier AUC measures separability, not causal contribution. A causal dataset-versus-representation decomposition would require controlled acquisition matching, representation interventions, or prospective data—not another interpretation of these same predictions.
