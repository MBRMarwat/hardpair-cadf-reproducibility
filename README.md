# HardPair-Net / CADF-Net reproducibility release

This repository snapshot supports the manuscript's reported HAM10000, BCN20000, HardPair-Net, CADF-Net v3, post-hoc audit, and final same-condition comparator results. It intentionally excludes original dermoscopic images and patient data.

Public repository: <https://github.com/MBRMarwat/hardpair-cadf-reproducibility>

Current license-updated submission release: `v1.0.1-submission`

## Data

- HAM10000: <https://doi.org/10.7910/DVN/DBW86T>; descriptor <https://doi.org/10.1038/sdata.2018.161>.
- BCN20000: <https://doi.org/10.6084/m9.figshare.24140028>; descriptor <https://doi.org/10.1038/s41597-024-03387-w>; CC BY 4.0.

Place the HAM10000 image directories and BCN20000 training image directory at the paths shown in the supplied YAML configurations, or edit only the path fields for your machine. Do not regenerate `data_manifests/ham10000_splits.csv` when reproducing the reported results.

## Environment

Create the environment from `environment.yml`, activate it, and run commands from the project root. The executed Windows environment used Python 3.10.20, PyTorch 2.2.2+cu118, torchvision 0.17.2+cu118, and CUDA on an NVIDIA GTX 1050.

## Main workflows

- `experiment_parts/part_01_ham10000_core_experiment/notebooks/HardPairNet_Lite.ipynb`: original HardPair-Net experiment, cell by cell.
- `experiment_parts/part_07_cadf_net_v3/notebooks/CADF_Net_v3_Experiments.ipynb`: CADF-Net v3 staged experiment, cell by cell.
- `experiment_parts/part_07_cadf_net_v3/V3_Audit/run_v3_audit.py`: read-only CADF checkpoint audit.
- `experiment_parts/part_09_final_presubmission_check/code/run_final_presubmission_check.py`: paired lesion-clustered intervals and implementation verification.
- `experiment_parts/part_09_final_presubmission_check/code/run_efficientnet_b0_comparator.py`: resumable same-split EfficientNet-B0 comparator.

Every training workflow writes an atomic latest checkpoint at least hourly and at epoch boundaries. Rerunning a completed comparator returns the saved checkpoint/results rather than retraining or reevaluating the test set.

## Result provenance

The organized `experiment_parts/*/analysis_and_results` directories contain the derived CSV/JSON/PNG evidence used in the tables and figures. The final mapping from numeric claims to source files is `experiment_parts/part_09_final_presubmission_check/analysis_and_results/claim_traceability.json`. Original image datasets are not redistributed. Selected CADF and comparator checkpoints are present so the audits can run; the complete checkpoint collection remains in the manuscript evidence ZIP.

## License

Except where otherwise noted, original material in this repository is licensed under the [Creative Commons Attribution 4.0 International license](LICENSE). Third-party software, pretrained weights, source datasets, and dataset images remain subject to their original licenses and terms. This repository does not relicense HAM10000 or BCN20000 images.

## Known limitations

- The HAM split is lesion-disjoint; patient-disjointness is not verifiable from the supplied metadata.
- CADF Stage 1/2 cached-feature validation had dropout active. Use the deterministic post-hoc checkpoint results.
- CADF A2 and A5 are smoke tests only.
- Later audits are exploratory because the held-out cohorts had already been examined.

Before publication, replace author placeholders in the manuscript statements and archive the license-updated release with a permanent DOI.

