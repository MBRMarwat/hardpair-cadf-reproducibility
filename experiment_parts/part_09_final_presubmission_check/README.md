# Final pre-submission check

This directory closes the four bounded tasks in the final checklist without altering any previously trained HardPair-Net or CADF-Net model.

## Completed work

1. `paired_correction_bootstrap.*` supplies 100,000-resample lesion-clustered paired confidence intervals for the fixed 87-image and shared 73-image CADF subsets, plus beneficial/harmful discordant counts.
2. `implementation_verification.json` checks the executed CADF code, configuration, split manifest, and selected checkpoints against the documented method.
3. `novelty_reference_verification.md` separates established ideas from the defensible new contribution and audits the 21 manuscript references.
4. `submission_readiness.md`, `claim_traceability.json`, and `manuscript_ready_additions.md` provide the reproducibility, reporting, and declaration material.
5. `comparator_full` contains one deliberately bounded same-split EfficientNet-B0 comparator. It was selected using validation data, calibrated on the fusion split, and evaluated once on the held-out test split.

## Important limits

- The new EfficientNet-B0 run is a strong same-condition architecture comparator, not a reproduction of every published state-of-the-art method.
- The held-out HAM10000 and BCN20000 cohorts have already been examined. All later analyses are diagnostic/exploratory.
- Patient-disjointness cannot be verified because the supplied HAM10000 metadata do not include patient identifiers.
- Pattern Analysis and Applications requires an accessible, unarchived code/data repository. A local ZIP alone does not satisfy that requirement; the author must publish the prepared repository material and insert the permanent URL.
- Author names, affiliations, ORCIDs, funding, competing interests, institutional ethics wording, and the final public repository URL require author input.

## Reproduction

Run with the `histo` environment from the project root:

```powershell
& 'C:\Users\Bilal Rehman\miniconda3\envs\histo\python.exe' `
  'experiment_parts\part_09_final_presubmission_check\code\run_final_presubmission_check.py'
```

The comparator is resumable from its latest checkpoint:

```powershell
& 'C:\Users\Bilal Rehman\miniconda3\envs\histo\python.exe' `
  'experiment_parts\part_09_final_presubmission_check\code\run_efficientnet_b0_comparator.py'
```

