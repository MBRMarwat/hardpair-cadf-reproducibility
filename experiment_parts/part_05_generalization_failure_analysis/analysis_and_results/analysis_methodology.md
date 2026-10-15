# Analysis methodology

## Frozen inputs

- HAM10000 held-out test probabilities and gate output from
  `outputs/full_20epochs/test_probabilities.pt`.
- Aligned HAM image/lesion IDs from `cached_predictions.pt`.
- BCN frozen predictions from the completed external evaluation.
- Saved ImageNet-pretrained 576-dimensional feature caches for both cohorts.
- HAM10000 and BCN20000 metadata plus local image files for summary statistics.

No diagnostic model predicts disease. No frozen disease model was executed.

## Metrics

MEL sensitivity is image-level unless explicitly marked lesion-level. Lesion
estimates average the seven saved probabilities within each MEL lesion before
argmax. Dataset differences use 2,000 independent lesion-level bootstrap
resamples. Wilson intervals describe individual stratified sensitivities.

The unrecoverable condition is `p(MEL)+p(NV) <= max(off-pair probability)`,
which means pair-restricted redistribution cannot make MEL a unique winner.

## Metadata harmonization

Age bins are `<40`, `40-59`, `60-79`, `80+`, and unknown. Sex is male,
female, or unknown. Sites are coarsened to head/neck, torso, upper extremity,
lower extremity, acral, oral/genital, or unknown. One-factor standardization
reweights BCN stratum-specific MEL sensitivity to the HAM MEL distribution.
It is descriptive, not causal.

## Image characteristics

Images are resized to 64x64 only for summary measurements: original width,
height, aspect ratio, megapixels, mean luminance, luminance contrast, HSV-like
saturation, mean adjacent-pixel edge strength, and center-minus-border
luminance. These do not reproduce a full image-quality assessment.

## Domain separability

Five-fold StratifiedGroupKFold keeps lesion IDs within folds. BCN samples are
diagnosis-matched to HAM counts; a separate analysis is MEL-only. Diagnostic
logistic regressions predict dataset (HAM vs BCN) from: geometry/resolution,
photometric thumbnail summaries, all basic image summaries,
ImageNet-pretrained features (standardized, 50-component PCA), or the frozen
baseline probability vector. These small CPU fits quantify separability only.
They are not medical model training and their AUCs are not causal shares.

Fine-tuned hidden feature vectors were not saved. The available representation
analysis therefore uses saved ImageNet-pretrained features and frozen decision
probabilities, not newly extracted fine-tuned activations.
