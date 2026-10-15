# HardPair-Net completed HAM10000 results package

This package was exported from frozen artifacts under `outputs/full_20epochs`
and `hardpair_diagnostics`. No checkpoint was modified. No network was
instantiated, trained, retuned, recalibrated, or run. No new model was evaluated
on the held-out test set. Tabular exports and plots are views of already-saved
metrics and probability arrays.

## Held-out test cohort

- Images: 1010
- Lesions: 747
- Class order: akiec, bcc, bkl, df, mel, nv, vasc
- Methods: baseline, auxiliary_only, fixed_fusion, learned_fusion

## File guide

- `confusion_matrices.csv/json/png`: seven-class held-out test confusion matrices.
- `per_class_metrics.csv/json`: precision, recall, F1, and support.
- `held_out_test_predictions.csv`: aligned image IDs, lesion IDs, labels,
  predictions, all seven probabilities for every method, and learned-gate alpha.
- `test_bootstrap_comparisons.csv`: the two paired lesion-level test comparisons
  saved by the original run, with 2,000-resample 95% intervals.
- `validation_bootstrap_comparisons.csv`: completed development/validation
  comparisons from the later diagnostic analysis; explicitly not test results.
- `epoch_training_history.csv/json` and `training_history.png`: all 20 epochs.
- `test_gate_*`, `test_prediction_*`, `unrecoverable_melanoma_counts.csv`, and
  gate PNGs: gate distributions and effects from saved outputs.
- `model_parameter_counts.csv/json`: counts derived from checkpoint tensor shapes.
- `compute_memory_and_timing.csv/json`: measured values, timestamp estimates,
  and unavailable fields with provenance.
- `source_artifact_checksums.json`: hashes of source artifacts used.
- `package_manifest.json`: hashes and sizes of every exported file.

## Availability caveats

The original run did not log exact stage-level training wall times, isolated
held-out test inference time, or full-training peak VRAM. These are marked
`unavailable`. Approximate training/prediction intervals derived earlier from
artifact timestamps are labeled `artifact_timestamp_estimate`, not measured.
The measured 0.09146 GiB value is only peak allocated CUDA memory during the
saved 100-step benchmark and must not be described as full-run peak VRAM.

The zero-width learned-versus-auxiliary pair-balanced-accuracy interval is the
saved original result: their pair correctness outcomes were identical for that
metric, even though the underlying probability arrays were not identical.
