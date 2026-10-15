# Novelty, comparator, and reference verification

Verification date: 2026-10-15.

## Defensible novelty statement

Dual heads, hierarchy-aware learning, late fusion, ensemble gating, and confusion-aware losses are established ideas and must not be claimed as novel. The defensible contribution is narrower:

1. an exact pair-mass-conserving probability redistribution in which the five off-pair class probabilities remain unchanged under primary MEL/NV fusion;
2. explicit recoverability and correction-envelope analysis showing which melanoma errors cannot be corrected under pair restriction and a coefficient cap;
3. a fixed-generalist-error-set, paired empirical analysis that distinguishes average specialist accuracy from useful complementary error correction; and
4. a cross-cohort audit showing that the available correction opportunities differ between the evaluated HAM10000 and BCN20000 cohorts, without claiming a causal explanation.

The paper should be framed as a controlled methodological and failure-analysis study, not as a state-of-the-art accuracy claim.

## Closest prior work

| Work | Established overlap | Distinction from this study | Verified source |
|---|---|---|---|
| HAC-LF | Hierarchy-aware contrastive learning and late fusion for skin lesions | Does not provide this study's pair-mass conservation proof, correction envelope, or fixed-error-set complementarity audit | [Hsu and Tseng, 2022](https://doi.org/10.1016/j.cmpb.2022.106666) |
| LightDPH | Lightweight hierarchy-aware dual projection heads; EfficientNet backbones | Projection heads are used for representation learning and removed downstream; not pair-restricted probability redistribution | [Hsu and Tseng, 2024](https://doi.org/10.1007/s41666-024-00174-5) |
| Confusion-aware loss | Directly targets class confusions in skin-lesion classification | Loss shaping is not the same as a constrained specialist fusion operator or a fixed-error-set correction analysis | [Su and Hamed, 2024](https://doi.org/10.21917/ijivp.2024.0482) |
| Prototypical decision tree | Hierarchical prediction and mistake-severity analysis | Uses prototypes and a decision tree on a different private clinical taxonomy; the article also has a published correction | [Yu et al., 2025](https://doi.org/10.1038/s41746-024-01395-z), [correction](https://doi.org/10.1038/s41746-025-01494-5) |
| MultiExCam | Heterogeneous ensemble with attention and gating on skin-lesion data | Confirms that adaptive gating/ensembling is established; it does not make this study's constrained correction-bound contribution | [Ruga et al., 2026](https://doi.org/10.1016/j.cmpb.2025.109081) |
| MHDF-Net | Hierarchical multimodal mixture of experts with dynamic gates | Uses images plus metadata on different datasets; confirms that dynamic gates are not novel | [Hu et al., 2026](https://doi.org/10.1016/j.neucom.2025.132585) |

Published scores from these studies are not numerically comparable to the present held-out results because datasets, taxonomies, splits, inputs, and training protocols differ.

## Same-condition comparator assessment

The previous package contained same-split HardPair generalist, auxiliary-only, fusion, simple adjustment, and operating-point comparisons. Those are adequate for the mechanistic claim but are not an independent modern-backbone comparison. One bounded EfficientNet-B0 experiment was therefore added using the exact frozen lesion split, 192-pixel preprocessing, augmentation, 20-epoch schedule, class-weighted loss, validation-only selection, fusion-split temperature calibration, and one held-out test evaluation.

| Model | Pair-balanced accuracy | MEL sensitivity | Macro-F1 |
|---|---:|---:|---:|
| Original HardPair generalist | 65.44% | 40.80% | 0.6186 |
| EfficientNet-B0 comparator | 64.95% | 37.60% | 0.6440 |

The lesion-level paired difference in pair-balanced accuracy (EfficientNet minus original generalist) was -0.61 percentage points, with 95% CI -6.57 to +5.10. Thus, the comparator improved macro-F1 but did not demonstrate a reliable pair-balanced advantage. This is a same-condition contemporary backbone benchmark; it is not a full reproduction of HAC-LF, LightDPH, or another published system.

## Audit of the 21 manuscript references

| Ref. | Verification result | Action |
|---:|---|---|
| 1 | Esteva et al., Nature 542, 115-118 (2017), DOI `10.1038/nature21056`: verified | Keep |
| 2 | Tschandl, Rosendahl, Kittler, Scientific Data 5, 180161 (2018), DOI `10.1038/sdata.2018.161`: verified | Keep; cite the dataset repository in Data Availability too |
| 3 | Mohammed, Afify, Hassanien, vol. 32(5), 2050036 (2020), DOI `10.4015/S1016237220500362`: verified | Keep only if discussed directly |
| 4 | Dandu, Jayakameshwaraiah, Ravi Kumar, vol. 33(1), 2050045 (2021), DOI `10.4015/S1016237220500453`: verified | Keep only if discussed directly |
| 5 | Howal and Wagh, vol. 37(5), 2550007 (2025), DOI `10.4015/S1016237225500073`: verified | Keep |
| 6 | Deepa and Madhavan, online 7 Oct 2025; vol. 38(3), 2026, DOI `10.4015/S1016237225500413`: verified | Preserve online/publication-year distinction |
| 7 | Kumar et al., breast-cancer ensemble study, DOI `10.4015/S1016237224500017`: verified but not a skin-lesion study | Remove unless it supports a specific general ensemble-method statement |
| 8 | Hsu and Tseng, CMPB 216, 106666 (2022), DOI `10.1016/j.cmpb.2022.106666`: verified | Keep |
| 9 | Su and Hamed, ICTACT 15(2), 3407-3410 (2024), DOI `10.21917/ijivp.2024.0482`: verified | Keep |
| 10 | Hsu and Tseng, JHIR 8, 619-639 (2024), DOI `10.1007/s41666-024-00174-5`: verified | Keep |
| 11 | Yu et al., npj Digital Medicine 8, 26 (2025), DOI `10.1038/s41746-024-01395-z`: verified | Add the correction DOI `10.1038/s41746-025-01494-5` |
| 12 | Lin et al., CBM 175, 108549 (2024), DOI `10.1016/j.compbiomed.2024.108549`: verified; full author list has six authors | `et al.` is acceptable; keep |
| 13 | Hu, Mei, Chen, Hou, CBM 176, 108594 (2024), DOI `10.1016/j.compbiomed.2024.108594`: verified | Keep |
| 14 | Gajera, Nayak, Zaveri, CBM 196(Pt A), 110729 (2025), DOI `10.1016/j.compbiomed.2025.110729`: verified | Keep |
| 15 | Hu, Zhang, Ying, Lang, ASC 185(Pt A), 113850 (2025), DOI `10.1016/j.asoc.2025.113850`: verified | Keep |
| 16 | Zhang, Li, Shi, Zhang, ASC 188, 114342; online 2025/issue 2026, DOI `10.1016/j.asoc.2025.114342`: verified | Preserve online/publication-year distinction |
| 17 | Howard et al., ICCV 2019, 1314-1324, DOI `10.1109/ICCV.2019.00140`: verified | Keep |
| 18 | Ruga, Caroprese, Vocaturo, Zumpano, CMPB 273, 109081 (2026), DOI `10.1016/j.cmpb.2025.109081`: verified | Keep |
| 19 | C. Hu et al., Neurocomputing 671, 132585 (2026), DOI `10.1016/j.neucom.2025.132585`: verified; the DOI record lists ten authors | `et al.` is acceptable; keep |
| 20 | Guo et al., ICML/PMLR 70, 1321-1330 (2017): verified | Keep |
| 21 | Saito and Rehmsmeier, PLOS ONE 10(3), e0118432 (2015), DOI `10.1371/journal.pone.0118432`: verified | Keep |

No DOI mismatch was found. References 6 and 16 need careful online-first versus issue-year formatting, reference 11 needs its correction cited, and reference 7 is topically peripheral.

