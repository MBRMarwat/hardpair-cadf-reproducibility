# CADF-Net v3 post-hoc audit

This directory contains a read-only audit of the completed CADF-Net v3 run. It does not retrain, tune, or overwrite any model. All generated files are written under `results/`.

The audit answers four questions:

1. Did specialist correction potential really fall from Stage 2 to Stage 3 when both stages are evaluated on the same images, definitions, and denominators?
2. Are the reported training-history values from the selected checkpoints or merely from the final epochs?
3. How often did learned alpha fusion and feasibility-aware beta transfer actually change a test prediction, and were those changes beneficial or harmful?
4. How large is the validation-to-test gap, including lesion-level bootstrap uncertainty?

## Run

From the Hard-Pair Research root in PowerShell:

```powershell
& 'C:\Users\Bilal Rehman\miniconda3\Scripts\conda.exe' run -n histo python 'experiment_parts\part_07_cadf_net_v3\V3_Audit\run_v3_audit.py'
```

The runner verifies all expected inputs, records SHA-256 checksums, uses deterministic evaluation transforms, caches checkpoint predictions with their source hashes, and validates probability conservation. Re-running it safely reuses matching inference caches and regenerates tables, figures, and the report.

## Main outputs

- `V3_Audit_Report.md`: research interpretation and verified findings.
- `results/audit_summary.json`: machine-readable headline findings.
- `results/implementation_observations.json`: the verified Stage 2 dropout-on validation defect and its impact.
- `results/stage_checkpoint_metrics.csv`: saved-record and identically re-evaluated checkpoint metrics.
- `results/stage2_stage3_paired_validation.csv`: image-level paired specialist/generalist states.
- `results/fixed_denominator_comparison.csv`: correction rates on fixed Stage 2, Stage 3, intersection, and union error sets.
- `results/test_gate_effects_per_image.csv`: per-image alpha/beta values, transfer amount, and prediction effects.
- `results/test_gate_change_summary.csv`: beneficial, harmful, wrong-to-wrong, and unchanged counts.
- `results/validation_test_comparison.csv` and `validation_test_gap_bootstrap.csv`: cohort comparison and lesion bootstrap intervals.
- `results/official_test_metric_reproduction.csv`: numerical check that reconstructed A0/A1/A3/A4 test metrics match the original export.
- `results/source_manifest.csv`: provenance and source hashes.

The locked definitions and analysis constraints are in `audit_plan.yaml`.

