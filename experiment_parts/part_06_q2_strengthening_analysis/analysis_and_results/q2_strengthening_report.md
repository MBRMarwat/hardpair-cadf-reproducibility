# HardPair-Net Q2-strengthening analysis

## Scientific status

The two scalar adjustments were fitted only on the original HAM fusion partition. HAM validation is the primary development comparison. The HAM test and BCN outcomes were already examined; new-rule results on those cohorts are exploratory and cannot be called independent confirmation.

## 1. Simpler comparators

The fitted MEL-logit offset is -0.0826; the pair-mass-preserving conditional log-odds offset is +0.0620. On validation, conditional adjustment has pair-balanced accuracy 0.6456, versus 0.6315 for fixed HardPair-Net, 0.6386 for learned HardPair-Net, and 0.6600 for the exploratory rebalanced-specialist fixed fusion. The plain baseline is 0.6485. Conditional adjustment MEL PR-AUC is 0.4572; because a scalar MEL-logit offset is monotone in baseline MEL probability, its threshold-free ranking curve is identical to baseline apart from numerical ties.

On previously examined BCN data, the conditional adjustment yields MEL sensitivity 22.8% and NV sensitivity 80.6%; these figures are exploratory.

Matched sensitivity/FPR tables compare both simple adjustments with the original and balanced-logistic fixed specialists. Across 44 predeclared lesion-bootstrap comparisons, 0 95% intervals exclude zero. These analyses test whether a specialist provides ranking information beyond one-parameter correction rather than relying only on argmax accuracy.

## 2. Theoretical correction limits

For HAM test MEL images, unrestricted pair redistribution gives an oracle sensitivity bound of 84.8%; the alpha<=0.75 ideal-specialist bound is 81.6%; and the existing specialist with an image-wise oracle gate is limited to 40.0%. The corresponding BCN bounds are 69.4%, 66.6%, and 23.7%.

The mutually exclusive categories connect the mathematical feasible region directly to observed errors. The actual-specialist oracle is optimistic and not implementable because it chooses alpha using the true label.

## 3. Cross-dataset error decomposition

Per-image and cross-tabulation files record pair ordering, pair-mass sufficiency, strongest off-pair class, specialist disagreement, and whether fixed/learned fusion corrected or worsened each melanoma. This remains descriptive, not causal.

## 4. Clinical trade-off

Decision curves compare MEL-versus-all net benefit with treat-all and treat-none. They are cohort-specific exploratory calculations; cohort prevalence and threshold weights are not asserted to represent deployment or patient outcomes. The accompanying count table reports additional false MEL predictions per additional correctly detected MEL whenever that ratio is defined.

## 5. Robustness controls

With diagnosis matching, one image per lesion, lesion-grouped CV, and features produced by identical 192x192 preprocessing, the domain classifier has mean ROC-AUC 1.000. Photometric-only AUC is 0.887. This shows persistent cohort separability after native geometry is removed but does not causally apportion performance loss.

The 93 perceptual-hash candidate pairs covering 44 BCN images have no byte-identical matches. Algorithmic identical-preprocessing review flags 0 pairs for closer review. Visual inspection of the 30 closest pairs found no obvious duplicate photograph; lesion shape, field, hair/background, or acquisition appearance differed. The remaining 63 pairs were not individually visually adjudicated. Algorithmic or visual screening is not proof of distinct provenance.

## 6. Independent confirmation

Unavailable: no third, genuinely independent labeled cohort was supplied. No claim of independent confirmation is made. Before any future cohort labels are inspected, freeze the selected rule, code, preprocessing, exclusions, and primary endpoint, and verify provenance against HAM10000 and BCN20000.
