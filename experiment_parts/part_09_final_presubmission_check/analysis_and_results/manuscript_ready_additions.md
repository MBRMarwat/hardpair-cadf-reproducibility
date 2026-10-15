# Manuscript-ready additions

These paragraphs are drafts. The authors must integrate them with the manuscript's terminology and verify every cross-reference.

## Methods: final paired Stage 2/Stage 3 analysis

To compare specialist complementarity before and after joint fine-tuning without changing the denominator, we evaluated the selected Stage 2 and Stage 3 checkpoints on two prespecified paired subsets: the 87 images misclassified by the Stage 2 generalist within the conditional melanoma-nevus decision, and the intersection of 73 images misclassified conditionally by both generalists. For each subset, the estimand was the image-level difference in specialist correctness (Stage 3 minus Stage 2). Ninety-five percent confidence intervals were obtained from 100,000 nonparametric bootstrap resamples of lesions with replacement; all images belonging to a sampled lesion were retained. Beneficial and harmful discordant changes were also counted. An exact image-level discordance test was retained only as a descriptive sensitivity analysis because it does not account for lesion clustering.

## Results: final paired Stage 2/Stage 3 analysis

On the fixed 87-image Stage 2 conditional-error set (60 lesions), specialist correctness decreased from 43/87 (49.43%) to 36/87 (41.38%), a difference of -8.05 percentage points (lesion-clustered 95% CI -20.00 to 4.40). Twenty-seven corrections were retained, 16 were lost, 9 were gained, and 35 images remained incorrect. On the shared 73-image error intersection (50 lesions), specialist correctness decreased from 35/73 (47.95%) to 26/73 (35.62%), a difference of -12.33 percentage points (95% CI -25.71 to 1.33). Twenty corrections were retained, 15 were lost, 6 were gained, and 32 remained incorrect. Both intervals included zero. These results indicate a directional reduction in complementarity after fine-tuning but do not establish a statistically definitive degradation or its cause.

## Methods: same-condition architecture comparator

To provide a bounded same-condition architecture comparison, we trained an ImageNet-pretrained EfficientNet-B0 seven-class classifier on the frozen HAM10000 lesion-disjoint split used by HardPair-Net. Images used the same 192-pixel aspect-preserving resize/pad operation, ImageNet normalization, horizontal and vertical flips, rotations, and color jitter. The final three feature modules and classifier were optimized for 20 epochs with class-weighted cross-entropy, AdamW, cosine learning-rate decay, and fixed seed 42. The checkpoint was selected by validation macro-F1 with melanoma sensitivity as a tie-breaker. A scalar temperature was then fitted on the separate fusion split. The held-out test set was evaluated once after all training, selection, and calibration choices were frozen. The pair-balanced difference against the original HardPair generalist was assessed by a 2,000-resample lesion-level paired bootstrap.

## Results: same-condition architecture comparator

The selected EfficientNet-B0 checkpoint occurred at epoch 19 (validation macro-F1 0.6512; validation melanoma sensitivity 47.46%). On the 1,010-image held-out test set, EfficientNet-B0 obtained pair-balanced accuracy 64.95%, melanoma sensitivity 37.60%, and macro-F1 0.6440. The corresponding original HardPair generalist values were 65.44%, 40.80%, and 0.6186. The lesion-level paired difference in pair-balanced accuracy (EfficientNet minus original generalist) was -0.61 percentage points (95% CI -6.57 to 5.10). Thus, EfficientNet-B0 improved macro-F1 but did not demonstrate a reliable advantage in melanoma-nevus pair-balanced accuracy. This result is a controlled backbone comparator and should not be presented as a direct reproduction of published hierarchy-aware systems trained under other protocols.

## Methods: implementation clarification

CADF-Net v3 accepts five-channel tensors in its shared encoder. The generalist view consists of normalized RGB concatenated with two zero channels. The specialist view consists of a deterministic label-free lesion crop represented by normalized RGB plus two color-decomposition channels. Because segmentation masks were unavailable, the crop is produced by a heuristic localizer and must not be described as the output of a learned segmentation model. Melanoma and nevus correspond to class indices 4 and 5, respectively. The learned primary gate is bounded by a sigmoid-transformed trainable cap initialized at 0.75. Secondary transfer is limited by the strongest off-pair margin and by 0.30 times the strongest off-pair probability before feasibility, alpha, and melanoma-preference gating.

## Limitations: validation defect and analysis status

The Stage 1 and Stage 2 cached-feature validation helper disabled gradients but did not switch the heads to evaluation mode, leaving dropout active during the recorded validation passes and checkpoint selection. Deterministic post-hoc evaluation of the saved checkpoints is therefore reported in place of the stochastic logged Stage 2 correction rate. Stage 3 validation used evaluation mode and was reproducible. The fixed-error-set analysis is exploratory: it diagnoses saved checkpoints after the HAM10000 test set and BCN20000 cohort had already been examined, and no new method has been confirmed on a third untouched cohort. Patient-disjointness also could not be verified because patient identifiers were not available in the supplied HAM10000 metadata.

