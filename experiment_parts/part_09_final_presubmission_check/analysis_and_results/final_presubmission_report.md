# Final pre-submission report

## Decision

All four scientific/reproducibility tasks in the supplied checklist are complete. No material arithmetic or fusion-implementation error was found. One additional same-split comparator was justified by the target journal's comparison requirement and was completed without modifying or retraining the existing HardPair-Net/CADF-Net models.

The work is scientifically ready to freeze. It is not yet ready to upload to the journal because the public unarchived repository URL, complete manuscript sections, and author-specific declarations are still missing.

## New statistical result

- Fixed Stage 2 error set: 43/87 versus 36/87; Stage 3 minus Stage 2 = -8.05 percentage points, lesion-clustered 95% CI -20.00 to 4.40; 9 gained and 16 lost corrections.
- Shared error intersection: 35/73 versus 26/73; difference = -12.33 points, 95% CI -25.71 to 1.33; 6 gained and 15 lost corrections.
- Both intervals cross zero. Report a directional concern, not proof that fine-tuning causes deterioration.

## New same-condition comparator

- EfficientNet-B0 was trained for 20 epochs on the frozen lesion-disjoint split and selected at epoch 19 using validation macro-F1.
- Test results: pair-balanced accuracy 64.95%, melanoma sensitivity 37.60%, macro-F1 0.6440.
- Original HardPair generalist: 65.44%, 40.80%, 0.6186.
- Paired lesion bootstrap for pair-balanced accuracy: -0.61 percentage points, 95% CI -6.57 to 5.10.
- Interpretation: EfficientNet improved macro-F1 but did not reliably improve the targeted MEL/NV balance.

## Implementation findings that must appear in the manuscript

1. CADF uses a five-channel shared encoder; the generalist receives RGB plus two zeros.
2. The specialist's lesion crop is heuristic localization, not learned segmentation.
3. The error-focused specialist loss, class mapping, learned alpha cap, and bounded transfer formula match the executed code.
4. Stage 1/2 cached-feature validation left dropout active. Use deterministic audit metrics and label this defect.
5. A2 and A5 were smoke-tested only; do not report them as completed full ablations.
6. The split is lesion-disjoint; patient-disjointness is unverified.

## Final claim boundary

The evidence supports a paper about limits of specialist complementarity and constrained probability fusion. It does not support claims that CADF-Net improves classification, that joint fine-tuning definitively causes the complementarity reduction, that acquisition shift caused the BCN sensitivity decline, or that the method is state of the art.

## Remaining non-experimental blockers

- assemble Methods, Results, Discussion, Conclusion, abstract, title page, captions, and declarations in the final manuscript;
- publish the reproducibility directory as accessible, unarchived files and add its URL/DOI;
- supply author, funding, competing-interest, ethics, contribution, and license information;
- cite the correction to Yu et al.; remove or justify the breast-cancer reference; and preserve online-first year distinctions.

No further experiments are recommended before submission unless these checks uncover a new validity problem.

