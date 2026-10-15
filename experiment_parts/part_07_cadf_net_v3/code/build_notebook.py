from __future__ import annotations

import json
from pathlib import Path


PART_DIR = Path(__file__).resolve().parents[1]
DESTINATION = PART_DIR / "notebooks" / "CADF_Net_v3_Experiments.ipynb"
_cell_number = 0


def next_cell_id() -> str:
    global _cell_number
    _cell_number += 1
    return f"cadf-v3-{_cell_number:02d}"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": next_cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# CADF-Net v3: staged, resumable experiments

Use the **Python (histo)** kernel. The notebook is intentionally split into stages so you can work according to your available time.

Every training stage writes an atomic checkpoint at each epoch boundary and at least once per hour during long epochs. Re-running an interrupted stage cell resumes from its latest checkpoint. Feature/output extraction is also resumable.

Scientific guardrail: the fusion partition is used to fit feasibility/gating, validation is used for selection, and the held-out test cell should be executed only after Stages 1–5 are fixed. The 100-image smoke results are engineering checks, not research findings."""
    ),
    markdown("## 1. Locate Part 7 and import the implementation"),
    code(
        """from pathlib import Path
import json, sys, torch

candidates = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd() / 'experiment_parts' / 'part_07_cadf_net_v3',
]
PART_DIR = next(
    path.resolve() for path in candidates
    if (path / 'configuration' / 'cadfnet_v3.yaml').is_file()
)
sys.path.insert(0, str(PART_DIR / 'code'))

from cadfnet_v3.checkpoint import atomic_torch_save
from cadfnet_v3.config import load_config
from cadfnet_v3.pipeline import (
    export_evaluation, prepare_run, preflight_check,
    run_external_evaluation, run_trainable_ablation,
)
from cadfnet_v3.training import (
    cache_initial_features, collect_outputs, evaluate_ablation_variants,
    initialize_components, train_stage1_generalist,
    train_stage2_specialist, train_stage3_finetune,
    train_stage4_feasibility, train_stage5_gate, write_histories,
)
print('Part directory:', PART_DIR)
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"""
    ),
    markdown("## 2. Choose full or smoke configuration"),
    code(
        """# Leave this False for the complete experiment. Set True only for a fresh engineering check.
SMOKE = False

config_path = PART_DIR / 'configuration' / 'cadfnet_v3.yaml'
override_path = PART_DIR / 'configuration' / 'smoke_overrides.yaml' if SMOKE else None
config = load_config(config_path, override_path)
frame, output_dir, device = prepare_run(config)

print('Mode:', 'SMOKE (100 images)' if SMOKE else 'FULL HAM10000')
print('Device:', device)
print('Output:', output_dir)
display(frame.groupby(['split', 'label']).size().unstack(fill_value=0))"""
    ),
    markdown(
        """## 3. Initialize the model and run preflight checks

The shared encoder always receives five channels. The generalist view is normalized RGB plus two zero channels; the specialist view is lesion-crop RGB plus melanin and hemoglobin channels."""
    ),
    code(
        """model, feasibility_predictor, gate, fusion = initialize_components(config)
preflight = preflight_check(model, frame, config, device)
print(json.dumps(preflight, indent=2))"""
    ),
    markdown(
        """## 4. Cache deterministic frozen-encoder features

This cache supports Stages 1–2. It checkpoints extraction progress hourly. Re-running this cell reuses a completed cache or resumes an interrupted one."""
    ),
    code(
        """head_frame = frame[frame['split'].isin(['train', 'val'])].reset_index(drop=True)
feature_cache = cache_initial_features(
    model, head_frame, output_dir / 'initial_feature_cache.pt', config, device
)
print({key: tuple(value.shape) for key, value in feature_cache.items() if torch.is_tensor(value)})"""
    ),
    markdown("## 5. Stage 1 — train the seven-class generalist head (15 full-run epochs)"),
    code(
        """histories = {}
histories['stage1'] = train_stage1_generalist(
    model, feature_cache, output_dir, config, device
)
histories['stage1'][-1]"""
    ),
    markdown("## 6. Stage 2 — train the MEL/NV specialist head with error-focused loss (15 epochs)"),
    code(
        """histories['stage2'] = train_stage2_specialist(
    model, feature_cache, output_dir, config, device
)
histories['stage2'][-1]"""
    ),
    markdown(
        """## 7. Stage 3 — fine-tune the last three encoder blocks (10 epochs)

This is the most computationally expensive stage. Re-running after interruption resumes from the last hourly/epoch checkpoint."""
    ),
    code(
        """train_frame = frame[frame['split'] == 'train'].reset_index(drop=True)
validation_frame = frame[frame['split'] == 'val'].reset_index(drop=True)
histories['stage3'] = train_stage3_finetune(
    model, train_frame, validation_frame, output_dir, config, device
)
histories['stage3'][-1]"""
    ),
    markdown(
        """## 8. Cache fusion-fitting and validation outputs

The held-out test split is deliberately not touched here."""
    ),
    code(
        """fusion_frame = frame[frame['split'] == 'fusion'].reset_index(drop=True)
fusion_outputs = collect_outputs(
    model, fusion_frame, output_dir / 'fusion_outputs.pt', config, device
)
validation_outputs = collect_outputs(
    model, validation_frame, output_dir / 'validation_outputs.pt', config, device
)
print('Fusion images:', len(fusion_outputs['labels']), '| Validation images:', len(validation_outputs['labels']))"""
    ),
    markdown("## 9. Stage 4 — train the transfer-help feasibility predictor (50 epochs)"),
    code(
        """histories['stage4'] = train_stage4_feasibility(
    feasibility_predictor, fusion_outputs, validation_outputs,
    output_dir, config, device
)
histories['stage4'][-1]"""
    ),
    markdown("## 10. Stage 5 — train the differentiable alpha/beta gate (30 epochs)"),
    code(
        """histories['stage5'] = train_stage5_gate(
    feasibility_predictor, gate, fusion,
    fusion_outputs, validation_outputs, output_dir, config, device
)
write_histories(output_dir, histories)
histories['stage5'][-1]"""
    ),
    markdown("## 11. Freeze the selected model bundle"),
    code(
        """atomic_torch_save({
    'model_state': model.state_dict(),
    'feasibility_state': feasibility_predictor.state_dict(),
    'gate_state': gate.state_dict(),
    'config': config,
}, output_dir / 'cadfnet_v3_final.pt')
print('Saved:', output_dir / 'cadfnet_v3_final.pt')"""
    ),
    markdown(
        """## 12. Final held-out HAM10000 test evaluation — run once

Run this only after the architecture, losses, hyperparameters, and Stages 1–5 are fixed. It evaluates A0, A1, A3, and A4 without additional training."""
    ),
    code(
        """test_frame = frame[frame['split'] == 'test'].reset_index(drop=True)
test_outputs = collect_outputs(
    model, test_frame, output_dir / 'test_outputs.pt', config, device
)
test_metrics, test_variants = evaluate_ablation_variants(
    test_outputs, feasibility_predictor, gate, fusion, config, device
)
export_evaluation(output_dir, test_outputs, test_metrics, test_variants, config)
display({name: {
    'pair_balanced_accuracy': values['pair_balanced_accuracy'],
    'mel_sensitivity': values['mel_sensitivity'],
    'nv_sensitivity': values['nv_sensitivity'],
    'macro_f1': values['macro_f1'],
} for name, values in test_metrics.items()})"""
    ),
    markdown(
        """## 13. Optional retraining ablation A2 — standard specialist CE

This reuses the frozen Stage-1 generalist and writes to `ablations/A2`. It has independent hourly checkpoints."""
    ),
    code(
        """RUN_A2 = False
if RUN_A2:
    a2_result = run_trainable_ablation(config, 'A2')
    display(a2_result['metrics'])
else:
    print('A2 skipped. Set RUN_A2=True when you have time.')"""
    ),
    markdown(
        """## 14. Optional retraining ablation A5 — no dual view

This reuses the frozen Stage-1 generalist and writes to `ablations/A5`. It has independent hourly checkpoints."""
    ),
    code(
        """RUN_A5 = False
if RUN_A5:
    a5_result = run_trainable_ablation(config, 'A5')
    display(a5_result['metrics'])
else:
    print('A5 skipped. Set RUN_A5=True when you have time.')"""
    ),
    markdown(
        """## 15. Optional frozen BCN20000 evaluation

Run only after the final bundle is frozen. No BCN labels are used for fitting. SCC is excluded from the primary seven-class comparison and reported in the status file."""
    ),
    code(
        """RUN_BCN_EXTERNAL = False
if RUN_BCN_EXTERNAL:
    external_result = run_external_evaluation(config)
    display(external_result['metrics'])
else:
    print('BCN evaluation skipped. Set RUN_BCN_EXTERNAL=True after HAM decisions are fixed.')"""
    ),
    markdown(
        """## Checkpoint behavior

- `*_latest.pt`: resume state, updated hourly and at every epoch boundary.
- `*_best.pt`: best validation-selected state for that stage.
- `*.progress.pt`: temporary resumable extraction state.
- Completed cells are safe to rerun; completed artifacts are loaded instead of recomputed.

Do not delete `latest`, `best`, cache, or progress files if you want to resume."""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (histo)",
            "language": "python",
            "name": "histo",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
DESTINATION.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(DESTINATION)
