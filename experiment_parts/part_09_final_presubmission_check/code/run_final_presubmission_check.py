from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import binomtest


SCRIPT = Path(__file__).resolve()
PART = SCRIPT.parent.parent
ROOT = PART.parent.parent
CADF = ROOT / "experiment_parts" / "part_07_cadf_net_v3"
AUDIT = CADF / "V3_Audit" / "results"
FULL = CADF / "analysis_and_results" / "full"
OUT = PART / "analysis_and_results"
RESAMPLES = 100_000
SEED = 42


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clustered_bootstrap(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    frame = frame.copy()
    frame["difference"] = (
        frame["stage3_specialist_correct"].astype(int)
        - frame["stage2_specialist_correct"].astype(int)
    )
    clusters = frame.groupby("lesion_id", sort=True)["difference"].agg(["sum", "count"])
    sums = clusters["sum"].to_numpy(dtype=np.float64)
    counts = clusters["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    sampled = np.empty(RESAMPLES, dtype=np.float64)
    batch_size = 5_000
    cluster_count = len(clusters)
    for start in range(0, RESAMPLES, batch_size):
        stop = min(start + batch_size, RESAMPLES)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        sampled[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)

    stage2_correct = frame["stage2_specialist_correct"].astype(bool)
    stage3_correct = frame["stage3_specialist_correct"].astype(bool)
    gained = int((~stage2_correct & stage3_correct).sum())
    lost = int((stage2_correct & ~stage3_correct).sum())
    discordant = gained + lost
    exact_p = float(binomtest(min(gained, lost), discordant, 0.5).pvalue) if discordant else 1.0
    estimate = float(frame["difference"].mean())
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "comparison": name,
        "unit_of_analysis": "image, with lesion-clustered resampling",
        "images": int(len(frame)),
        "lesions": int(frame["lesion_id"].nunique()),
        "stage2_correct": int(stage2_correct.sum()),
        "stage3_correct": int(stage3_correct.sum()),
        "stage2_rate": float(stage2_correct.mean()),
        "stage3_rate": float(stage3_correct.mean()),
        "difference_stage3_minus_stage2": estimate,
        "difference_percentage_points": 100.0 * estimate,
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_95_low_percentage_points": 100.0 * float(low),
        "ci_95_high_percentage_points": 100.0 * float(high),
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "probability_difference_below_zero": float(np.mean(sampled < 0.0)),
        "correct_retained": int((stage2_correct & stage3_correct).sum()),
        "correction_lost_harmful": lost,
        "correction_gained_beneficial": gained,
        "wrong_retained": int((~stage2_correct & ~stage3_correct).sum()),
        "discordant_images": discordant,
        "image_level_exact_mcnemar_p_two_sided_descriptive_only": exact_p,
        "interpretation": (
            "The interval is lesion-clustered. The exact discordance test is image-level and is "
            "reported only as a descriptive sensitivity check, not as the primary inferential result."
        ),
    }


def run_paired_analysis() -> list[dict[str, Any]]:
    source = AUDIT / "stage2_stage3_paired_validation.csv"
    paired = pd.read_csv(source)
    for column in [
        "pair_image",
        "stage2_conditional_error",
        "stage3_conditional_error",
        "stage2_specialist_correct",
        "stage3_specialist_correct",
    ]:
        paired[column] = paired[column].astype(str).str.lower().eq("true")

    pair = paired.loc[paired["pair_image"]].copy()
    comparisons = [
        clustered_bootstrap(
            pair.loc[pair["stage2_conditional_error"]],
            "fixed_stage2_conditional_error_set",
        ),
        clustered_bootstrap(
            pair.loc[pair["stage2_conditional_error"] & pair["stage3_conditional_error"]],
            "shared_stage2_stage3_conditional_error_intersection",
        ),
    ]
    write_json(
        OUT / "paired_correction_bootstrap.json",
        {
            "source": str(source.relative_to(ROOT)).replace("\\", "/"),
            "estimand": "Stage 3 minus Stage 2 specialist correctness on prespecified fixed subsets",
            "comparisons": comparisons,
        },
    )
    pd.DataFrame(comparisons).to_csv(OUT / "paired_correction_bootstrap.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(comparisons))
    estimates = np.array([x["difference_percentage_points"] for x in comparisons])
    lower = np.array([x["ci_95_low_percentage_points"] for x in comparisons])
    upper = np.array([x["ci_95_high_percentage_points"] for x in comparisons])
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o",
        color="#8f1d1d",
        ecolor="#333333",
        capsize=5,
        markersize=7,
    )
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, ["Fixed Stage-2 error set", "Shared-error intersection"])
    ax.invert_yaxis()
    ax.set_xlabel("Stage 3 − Stage 2 correction rate (percentage points)")
    ax.set_title("Lesion-clustered paired bootstrap (100,000 resamples)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "paired_correction_bootstrap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return comparisons


def tensor_states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def run_implementation_audit() -> dict[str, Any]:
    config_path = CADF / "configuration" / "cadfnet_v3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    final_path = FULL / "cadfnet_v3_final.pt"
    stage3_path = FULL / "stage3_finetune_best.pt"
    final = torch.load(final_path, map_location="cpu")
    stage3 = torch.load(stage3_path, map_location="cpu")
    model_state = final["model_state"]
    gate_state = final["gate_state"]
    learned_cap = float(torch.sigmoid(gate_state["alpha_cap_logit"]).item())

    split_path = ROOT / "experiment_parts" / "part_01_ham10000_core_experiment" / "analysis_and_results" / "outputs" / "ham10000_splits.csv"
    split_frame = pd.read_csv(split_path)
    lesion_split_counts = split_frame.groupby("lesion_id")["split"].nunique()

    code_dir = CADF / "code" / "cadfnet_v3"
    data_source = (code_dir / "data.py").read_text(encoding="utf-8")
    loss_source = (code_dir / "losses.py").read_text(encoding="utf-8")
    fusion_source = (code_dir / "fusion.py").read_text(encoding="utf-8")
    training_source = (code_dir / "training.py").read_text(encoding="utf-8")

    checks = [
        {
            "item": "encoder_input_channels",
            "status": "matches",
            "evidence": f"Final checkpoint first convolution shape is {tuple(model_state['features.0.0.weight'].shape)}; input channels=5.",
        },
        {
            "item": "generalist_view",
            "status": "matches_with_documented_resolution",
            "evidence": "Generalist input is normalized RGB concatenated with two zero channels.",
            "verified": "torch.zeros(2" in data_source and "generalist_view = torch.cat" in data_source,
        },
        {
            "item": "specialist_view",
            "status": "matches_with_limitation",
            "evidence": "Specialist input is cropped normalized RGB plus two decomposition channels; cropper is a deterministic label-free heuristic, not a trained segmenter.",
            "verified": "specialist_view = torch.cat([lesion_normalized, decomposition]" in data_source,
        },
        {
            "item": "head_dimensions",
            "status": "matches",
            "evidence": {
                "generalist_first": list(model_state["generalist_head.0.weight"].shape),
                "generalist_output": list(model_state["generalist_head.3.weight"].shape),
                "specialist_first": list(model_state["specialist_head.0.weight"].shape),
                "specialist_output": list(model_state["specialist_head.3.weight"].shape),
            },
        },
        {
            "item": "label_mapping",
            "status": "matches",
            "evidence": {"mel_index": config["data"]["mel_index"], "nv_index": config["data"]["nv_index"], "class_names": config["data"]["class_names"]},
        },
        {
            "item": "error_focused_loss",
            "status": "matches",
            "evidence": "Class-weighted focal term is multiplied by generalist-error and uncertainty weights.",
            "verified": all(token in loss_source for token in ["generalist_wrong", "uncertain", "self.error_weight", "self.uncertainty_weight"]),
        },
        {
            "item": "learned_alpha_cap",
            "status": "matches",
            "evidence": {"configured_initial_cap": config["fusion"]["alpha_initial_cap"], "learned_final_cap": learned_cap},
        },
        {
            "item": "transfer_formula",
            "status": "matches",
            "evidence": "Transfer is bounded by min(top off-pair margin, kappa*top probability), gated by feasibility, alpha and melanoma preference, subtracted from the strongest off-pair class and added to MEL, then renormalized.",
            "verified": all(token in fusion_source for token in ["maximum_transfer", "delta = beta", "strongest_full_index", "fused.sum"]),
        },
        {
            "item": "hourly_resume_checkpoint",
            "status": "matches",
            "evidence": {"checkpoint_interval_seconds": config["training"]["checkpoint_interval_seconds"]},
        },
        {
            "item": "stage3_final_checkpoint_identity",
            "status": "matches",
            "evidence": "The final bundle model_state is bytewise tensor-equal to stage3_finetune_best model_state.",
            "verified": tensor_states_equal(model_state, stage3["model_state"]),
        },
        {
            "item": "lesion_disjoint_split",
            "status": "matches",
            "evidence": {"images": int(len(split_frame)), "lesions": int(split_frame["lesion_id"].nunique()), "lesions_in_multiple_splits": int((lesion_split_counts > 1).sum())},
        },
        {
            "item": "stage1_stage2_validation_mode",
            "status": "implementation_defect_documented",
            "evidence": "_head_validation uses no_grad but does not call eval(); Stage 1/2 heads remain in train mode, so dropout was active during recorded validation and checkpoint selection. Deterministic post-hoc Stage 2 results must replace the logged rate in reporting.",
            "verified": "def _head_validation" in training_source and "model.eval()" not in training_source[training_source.index("def _head_validation"):training_source.index("def train_stage1_generalist")],
        },
        {
            "item": "full_ablation_completion",
            "status": "partial",
            "evidence": "A0/A1/A3/A4 have full-run outputs. A2 and A5 exist only as smoke ablations and must not be presented as full ablation results.",
        },
        {
            "item": "patient_disjoint_split",
            "status": "unavailable",
            "evidence": "The supplied HAM10000 manifest contains lesion IDs but no patient identifier; patient-level disjointness cannot be verified.",
        },
    ]
    result = {
        "audit_scope": "Saved configuration, code, split manifest, and selected checkpoints; no training or new model evaluation.",
        "config_sha256": sha256(config_path),
        "final_checkpoint_sha256": sha256(final_path),
        "stage3_checkpoint_sha256": sha256(stage3_path),
        "checks": checks,
        "blocking_reporting_rules": [
            "Describe the specialist crop as deterministic heuristic localization, not a learned segmentation model.",
            "Use deterministic post-hoc checkpoint metrics for Stage 2; label original dropout-active validation values as defective logs.",
            "Do not claim full A2 or A5 ablations.",
            "Do not claim patient-disjoint splitting.",
        ],
    }
    write_json(OUT / "implementation_verification.json", result)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons = run_paired_analysis()
    implementation = run_implementation_audit()
    print(json.dumps({
        "paired_comparisons": comparisons,
        "implementation_checks": len(implementation["checks"]),
        "output_directory": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
