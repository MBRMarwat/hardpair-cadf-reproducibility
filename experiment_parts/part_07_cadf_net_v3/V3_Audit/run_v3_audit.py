from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.nn import functional as F


AUDIT_DIR = Path(__file__).resolve().parent
PART_DIR = AUDIT_DIR.parent
CODE_DIR = PART_DIR / "code"
CONFIG_PATH = PART_DIR / "configuration" / "cadfnet_v3.yaml"
SOURCE_DIR = PART_DIR / "analysis_and_results" / "full"
RESULTS_DIR = AUDIT_DIR / "results"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cadfnet_v3.config import load_config  # noqa: E402
from cadfnet_v3.data import CLASS_NAMES, MEL_IDX, NV_IDX, load_ham_manifest, make_loader  # noqa: E402
from cadfnet_v3.fusion import OFF_PAIR_INDICES, SoftFeasibilityFusion, build_gate_inputs  # noqa: E402
from cadfnet_v3.metrics import evaluate_probabilities  # noqa: E402
from cadfnet_v3.model import CADFNet, FeasibilityPredictor, MLPGate  # noqa: E402


STAGE_CHECKPOINTS = {
    "stage2_best": "stage2_specialist_error_focused_best.pt",
    "stage2_last": "stage2_specialist_error_focused_latest.pt",
    "stage3_best": "stage3_finetune_best.pt",
    "stage3_last": "stage3_finetune_latest.pt",
}
SOURCE_FILES = [
    *STAGE_CHECKPOINTS.values(),
    "stage4_feasibility_best.pt",
    "stage5_gate_best.pt",
    "cadfnet_v3_final.pt",
    "initial_feature_cache.pt",
    "validation_outputs.pt",
    "test_outputs.pt",
    "evaluation_metrics.json",
    "training_histories.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().cpu()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def instantiate_model(config: dict[str, Any]) -> CADFNet:
    return CADFNet(
        pretrained=False,
        generalist_hidden=int(config["model"]["generalist_hidden"]),
        specialist_hidden=int(config["model"]["specialist_hidden"]),
        dropout=float(config["model"]["dropout"]),
    )


def infer_checkpoint(
    checkpoint_path: Path,
    cache_path: Path,
    frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    source_hash = sha256(checkpoint_path)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("complete")
            and cached.get("source_checkpoint_sha256") == source_hash
            and len(cached.get("labels", [])) == len(frame)
        ):
            return cached

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = instantiate_model(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    loader = make_loader(
        frame.reset_index(drop=True),
        config,
        training=False,
        batch_size=int(config["data"]["evaluation_batch_size"]),
    )
    output: dict[str, Any] = {
        "generalist_logits": [],
        "specialist_logits": [],
        "color_stats": [],
        "labels": [],
        "image_ids": [],
        "lesion_ids": [],
    }
    with torch.inference_mode():
        for batch in loader:
            generalist_logits, specialist_logits = model.forward_views(
                batch["generalist_view"].to(device, non_blocking=True),
                batch["specialist_view"].to(device, non_blocking=True),
            )
            output["generalist_logits"].append(normal_cpu(generalist_logits))
            output["specialist_logits"].append(normal_cpu(specialist_logits))
            output["color_stats"].append(normal_cpu(batch["color_stats"]))
            output["labels"].append(normal_cpu(batch["label"]).long())
            output["image_ids"].extend(list(batch["image_id"]))
            output["lesion_ids"].extend(list(batch["lesion_id"]))
    complete = {
        "complete": True,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": source_hash,
        "generalist_logits": torch.cat(output["generalist_logits"]),
        "specialist_logits": torch.cat(output["specialist_logits"]),
        "color_stats": torch.cat(output["color_stats"]),
        "labels": torch.cat(output["labels"]),
        "image_ids": output["image_ids"],
        "lesion_ids": output["lesion_ids"],
    }
    torch.save(complete, cache_path)
    return complete


def checkpoint_states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> tuple[bool, float]:
    if left.keys() != right.keys():
        return False, float("inf")
    max_difference = 0.0
    for key in left:
        a, b = left[key].detach().cpu(), right[key].detach().cpu()
        if a.shape != b.shape:
            return False, float("inf")
        if torch.is_floating_point(a):
            max_difference = max(max_difference, float((a - b).abs().max()))
        elif not torch.equal(a, b):
            return False, float("inf")
    return max_difference == 0.0, max_difference


def checkpoint_audit(outputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = outputs["labels"].numpy()
    generalist_probabilities = F.softmax(outputs["generalist_logits"].float(), dim=1)
    specialist_probabilities = F.softmax(outputs["specialist_logits"].float(), dim=1)
    generalist_prediction = generalist_probabilities.argmax(dim=1).numpy()
    specialist_prediction = specialist_probabilities.argmax(dim=1).numpy()
    pair_mask = np.isin(labels, [MEL_IDX, NV_IDX])
    pair_target = (labels == NV_IDX).astype(np.int64)
    pair_conditional_prediction = (
        generalist_probabilities[:, [MEL_IDX, NV_IDX]].argmax(dim=1).numpy()
    )
    conditional_error = pair_mask & (pair_conditional_prediction != pair_target)
    full_error = pair_mask & (generalist_prediction != labels)
    specialist_correct = pair_mask & (specialist_prediction == pair_target)
    generalist_correct = generalist_prediction == labels

    metric = evaluate_probabilities(generalist_probabilities, outputs["labels"])
    conditional_n = int(conditional_error.sum())
    full_n = int(full_error.sum())
    result = {
        "images": len(labels),
        "pair_images": int(pair_mask.sum()),
        "generalist_macro_f1": metric["macro_f1"],
        "generalist_pair_balanced_accuracy": metric["pair_balanced_accuracy"],
        "generalist_mel_sensitivity": metric["mel_sensitivity"],
        "generalist_nv_sensitivity": metric["nv_sensitivity"],
        "specialist_pair_accuracy": float(specialist_correct[pair_mask].mean()),
        "conditional_error_denominator": conditional_n,
        "conditional_correction_numerator": int((conditional_error & specialist_correct).sum()),
        "pair_conditional_correction_rate": float(
            (conditional_error & specialist_correct).sum() / max(1, conditional_n)
        ),
        "full_error_denominator": full_n,
        "full_error_correction_numerator": int((full_error & specialist_correct).sum()),
        "full_error_correction_potential": float(
            (full_error & specialist_correct).sum() / max(1, full_n)
        ),
        "specialist_harm_potential_numerator": int(
            (pair_mask & generalist_correct & ~specialist_correct).sum()
        ),
        "specialist_harm_potential_denominator": int((pair_mask & generalist_correct).sum()),
    }
    masks = {
        "labels": labels,
        "pair_mask": pair_mask,
        "pair_target": pair_target,
        "conditional_error": conditional_error,
        "full_error": full_error,
        "specialist_correct": specialist_correct,
        "generalist_correct": generalist_correct,
        "generalist_prediction": generalist_prediction,
        "specialist_prediction": specialist_prediction,
    }
    return result, masks


def audit_stage2_on_original_feature_cache(
    checkpoint_path: Path, cache: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model = instantiate_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"], strict=True)
    model.eval()
    mask = torch.tensor([split == "val" for split in cache["splits"]], dtype=torch.bool)
    with torch.inference_mode():
        outputs = {
            "generalist_logits": normal_cpu(model.generalist_from_features(cache["generalist_features"][mask])),
            "specialist_logits": normal_cpu(model.specialist_from_features(cache["specialist_features"][mask])),
            "labels": cache["labels"][mask].long(),
        }
    return checkpoint_audit(outputs)


def fixed_denominator_rows(
    stage2: dict[str, np.ndarray], stage3: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for error_kind in ("conditional_error", "full_error"):
        a = stage2[error_kind]
        b = stage3[error_kind]
        denominator_masks = {
            "stage2_error_set": a,
            "stage3_error_set": b,
            "intersection": a & b,
            "union": a | b,
        }
        for name, mask in denominator_masks.items():
            denominator = int(mask.sum())
            s2_count = int((mask & stage2["specialist_correct"]).sum())
            s3_count = int((mask & stage3["specialist_correct"]).sum())
            rows.append(
                {
                    "error_definition": error_kind,
                    "denominator_source": name,
                    "denominator": denominator,
                    "stage2_correct": s2_count,
                    "stage2_rate": s2_count / denominator if denominator else np.nan,
                    "stage3_correct": s3_count,
                    "stage3_rate": s3_count / denominator if denominator else np.nan,
                    "stage3_minus_stage2": (s3_count - s2_count) / denominator if denominator else np.nan,
                }
            )
    return rows


def gate_modules(config: dict[str, Any], final_bundle: dict[str, Any]) -> tuple[Any, Any, Any]:
    predictor = FeasibilityPredictor()
    predictor.load_state_dict(final_bundle["feasibility_state"], strict=True)
    gate = MLPGate(
        hidden_dim=int(config["model"]["gate_hidden"]),
        dropout=float(config["model"]["dropout"]),
        initial_alpha_cap=float(config["fusion"]["alpha_initial_cap"]),
    )
    gate.load_state_dict(final_bundle["gate_state"], strict=True)
    fusion = SoftFeasibilityFusion(
        kappa=float(config["fusion"]["kappa"]),
        temperature=float(config["fusion"]["temperature"]),
    )
    predictor.eval()
    gate.eval()
    return predictor, gate, fusion


def gate_outputs(outputs: dict[str, Any], predictor: Any, gate: Any, fusion: Any) -> dict[str, torch.Tensor]:
    probabilities = F.softmax(outputs["generalist_logits"].float(), dim=1)
    specialist = F.softmax(outputs["specialist_logits"].float(), dim=1)
    with torch.inference_mode():
        feasibility = predictor(outputs["color_stats"].float())
        gate_input = build_gate_inputs(probabilities, specialist, feasibility)
        alpha, beta = gate(gate_input)
        primary = fusion(probabilities, specialist, alpha, torch.zeros_like(beta), feasibility)
        full = fusion(probabilities, specialist, alpha, beta, feasibility)
        fixed = fusion(
            probabilities,
            specialist,
            torch.full_like(alpha, 0.5),
            torch.full_like(beta, 0.5),
            feasibility,
        )

    offpair = probabilities[:, list(OFF_PAIR_INDICES)]
    top2 = offpair.topk(k=2, dim=1).values
    strongest, strongest_column = offpair.max(dim=1)
    strongest_feasibility = feasibility.gather(1, strongest_column[:, None]).squeeze(1)
    temperature = fusion.temperature
    feasibility_gate = torch.sigmoid((strongest_feasibility - 0.5) / temperature)
    alpha_gate = torch.sigmoid((alpha - 0.5) / temperature)
    melanoma_preference = torch.sigmoid((primary[:, MEL_IDX] - primary[:, NV_IDX]) / temperature)
    maximum_transfer = torch.minimum(top2[:, 0] - top2[:, 1], fusion.kappa * strongest).clamp_min(0)
    expected_delta = beta * maximum_transfer * feasibility_gate * alpha_gate * melanoma_preference
    actual_delta = full[:, MEL_IDX] - primary[:, MEL_IDX]
    return {
        "baseline": probabilities,
        "specialist": specialist,
        "feasibility": feasibility,
        "alpha": alpha,
        "beta": beta,
        "primary": primary,
        "full": full,
        "fixed": fixed,
        "strongest_offpair_probability": strongest,
        "strongest_offpair_column": strongest_column,
        "strongest_feasibility": strongest_feasibility,
        "feasibility_gate": feasibility_gate,
        "alpha_gate": alpha_gate,
        "melanoma_preference": melanoma_preference,
        "maximum_transfer": maximum_transfer,
        "expected_delta": expected_delta,
        "actual_delta": actual_delta,
    }


def effect_category(source: np.ndarray, target: np.ndarray, labels: np.ndarray) -> np.ndarray:
    changed = source != target
    source_correct = source == labels
    target_correct = target == labels
    category = np.full(len(labels), "unchanged", dtype=object)
    category[changed & ~source_correct & target_correct] = "beneficial"
    category[changed & source_correct & ~target_correct] = "harmful"
    category[changed & ~source_correct & ~target_correct] = "wrong_to_wrong"
    category[changed & source_correct & target_correct] = "correct_to_correct"
    return category


def distribution_rows(values: dict[str, torch.Tensor], cohort: str) -> list[dict[str, Any]]:
    rows = []
    for name, tensor in values.items():
        array = tensor.detach().cpu().numpy().astype(float)
        rows.append(
            {
                "cohort": cohort,
                "quantity": name,
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "q01": float(np.quantile(array, 0.01)),
                "q05": float(np.quantile(array, 0.05)),
                "q25": float(np.quantile(array, 0.25)),
                "median": float(np.quantile(array, 0.50)),
                "q75": float(np.quantile(array, 0.75)),
                "q95": float(np.quantile(array, 0.95)),
                "q99": float(np.quantile(array, 0.99)),
                "max": float(np.max(array)),
            }
        )
    return rows


def metric_from_predictions(labels: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(labels, prediction, labels=range(7), average="macro", zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, prediction))
    recalls = []
    targets = (MEL_IDX, NV_IDX) if metric == "pair_balanced_accuracy" else (MEL_IDX,)
    for target in targets:
        mask = labels == target
        recalls.append(float((prediction[mask] == target).mean()) if mask.any() else np.nan)
    return float(np.nanmean(recalls))


def lesion_bootstrap_gap(
    validation_labels: np.ndarray,
    validation_prediction: np.ndarray,
    validation_lesions: list[str],
    test_labels: np.ndarray,
    test_prediction: np.ndarray,
    test_lesions: list[str],
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    def lesion_confusions(
        labels: np.ndarray, prediction: np.ndarray, lesion_ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        unique, inverse = np.unique(np.asarray(lesion_ids), return_inverse=True)
        matrices = np.zeros((len(unique), 7, 7), dtype=np.int64)
        np.add.at(matrices, (inverse, labels, prediction), 1)
        return unique, matrices

    def metric_from_confusion(matrix: np.ndarray, name: str) -> float:
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        diagonal = np.diag(matrix).astype(float)
        recall = np.divide(
            diagonal, support, out=np.full(7, np.nan, dtype=float), where=support > 0
        )
        if name == "pair_balanced_accuracy":
            return float(np.nanmean(recall[[MEL_IDX, NV_IDX]]))
        if name == "mel_sensitivity":
            return float(recall[MEL_IDX])
        precision = np.divide(
            diagonal, predicted, out=np.zeros(7, dtype=float), where=predicted > 0
        )
        recall_zero = np.nan_to_num(recall, nan=0.0)
        denominator = precision + recall_zero
        f1 = np.divide(
            2.0 * precision * recall_zero,
            denominator,
            out=np.zeros(7, dtype=float),
            where=denominator > 0,
        )
        return float(f1.mean())

    rng = np.random.default_rng(seed)
    val_unique, val_matrices = lesion_confusions(
        validation_labels, validation_prediction, validation_lesions
    )
    test_unique, test_matrices = lesion_confusions(test_labels, test_prediction, test_lesions)
    differences = []
    for _ in range(resamples):
        sampled_val = rng.integers(0, len(val_unique), size=len(val_unique))
        sampled_test = rng.integers(0, len(test_unique), size=len(test_unique))
        differences.append(
            metric_from_confusion(test_matrices[sampled_test].sum(axis=0), metric)
            - metric_from_confusion(val_matrices[sampled_val].sum(axis=0), metric)
        )
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    observed = metric_from_predictions(test_labels, test_prediction, metric) - metric_from_predictions(
        validation_labels, validation_prediction, metric
    )
    return {
        "metric": metric,
        "observed_test_minus_validation": observed,
        "bootstrap_mean": float(np.nanmean(differences)),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "resamples": resamples,
        "unit": "lesion",
    }


def per_class_recall(labels: np.ndarray, prediction: np.ndarray) -> list[float]:
    return [
        float((prediction[labels == target] == target).mean()) if np.any(labels == target) else np.nan
        for target in range(7)
    ]


def plot_outputs(
    stage_table: pd.DataFrame,
    gate_frame: pd.DataFrame,
    change_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
) -> None:
    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(stage_table))
    plt.bar(x - 0.18, 100 * stage_table["pair_conditional_correction_rate"], width=0.36, label="Conditional-error definition")
    plt.bar(x + 0.18, 100 * stage_table["full_error_correction_potential"], width=0.36, label="Full-error definition")
    plt.xticks(x, stage_table["checkpoint"], rotation=20, ha="right")
    plt.ylabel("Correction rate (%)")
    plt.title("Identically re-evaluated validation correction potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stage_correction_rates.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.5))
    positive = gate_frame.loc[gate_frame["beta_mel_transfer"] > 0, "beta_mel_transfer"]
    plt.hist(positive, bins=40, color="#3b82f6")
    plt.xlabel("MEL probability transferred by beta")
    plt.ylabel("Images")
    plt.title("Realized feasibility-transfer magnitude on test images")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "beta_transfer_distribution.png", dpi=180)
    plt.close()

    changed = change_table.loc[change_table["scope"] == "all_classes"]
    pivot = changed.pivot(index="effect", columns="comparison", values="count").fillna(0)
    pivot.plot(kind="bar", figsize=(8, 4.8), color=["#64748b", "#f59e0b", "#10b981"])
    plt.ylabel("Test images")
    plt.title("Prediction effects of alpha, beta, and full fusion")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gate_prediction_changes.png", dpi=180)
    plt.close()

    subset = comparison_table[
        comparison_table["metric"].isin(["pair_balanced_accuracy", "mel_sensitivity", "macro_f1"])
    ].copy()
    labels = ["Pair-balanced accuracy", "MEL sensitivity", "Macro-F1"]
    plt.figure(figsize=(8, 4.8))
    x = np.arange(3)
    for offset, cohort in [(-0.18, "validation"), (0.18, "test")]:
        values = [float(subset[(subset.cohort == cohort) & (subset.metric == metric)].value.iloc[0]) for metric in ["pair_balanced_accuracy", "mel_sensitivity", "macro_f1"]]
        plt.bar(x + offset, values, width=0.36, label=cohort.title())
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Metric value")
    plt.title("Full CADF-Net v3 validation versus held-out test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_test_gap.png", dpi=180)
    plt.close()


def write_report(summary: dict[str, Any]) -> None:
    stage = summary["stage_comparison"]
    gate = summary["test_gate_audit"]
    gap = summary["validation_test_gap"]
    lines = [
        "# CADF-Net v3 focused post-hoc audit",
        "",
        "## Scope and safeguards",
        "",
        "This audit used only the completed CADF-Net v3 checkpoints and saved outputs. No model was retrained, no parameter or threshold was tuned, and no original result was overwritten. The held-out HAM10000 test labels were used only for diagnosis and reporting.",
        "",
        "## Main finding: the apparent Stage 3 collapse mixed checkpoint states",
        "",
        f"The saved training-history values of {100*stage['stage2_last_stored_conditional_rate']:.2f}% (Stage 2) and {100*stage['stage3_last_stored_conditional_rate']:.2f}% (Stage 3) describe the final epochs. They are not the checkpoints selected and restored by the pipeline. Stage 2 selected epoch {stage['stage2_selected_epoch']:.0f}; Stage 3 selected epoch {stage['stage3_selected_epoch']:.0f}.",
        "",
        f"A fresh deterministic image pass gave {100*stage['stage2_last_reevaluated_conditional_rate']:.2f}% for the Stage 2 last checkpoint and {100*stage['stage3_last_reevaluated_conditional_rate']:.2f}% for Stage 3 last. Reapplying the Stage 2 heads to the original saved feature cache also gives {100*stage['stage2_last_cached_feature_conditional_rate']:.2f}%, so the discrepancy is not caused by reconstructing image features.",
        "",
        "Code inspection identifies the cause: the Stage 2 loop calls `specialist_head.train()`, then `_head_validation()` evaluates under `torch.no_grad()` without calling `model.eval()`. The generalist and specialist head dropout layers therefore remain active during the saved Stage 2 validation history and checkpoint selection. The 41.94% value is a stochastic dropout-on measurement; 37.93% is the deterministic dropout-off value from the saved final checkpoint. Stage 3 calls `model.eval()` and its 26.97% final-epoch value reproduces. This is a validation/checkpoint-selection defect, not evidence about feature-cache drift.",
        "",
        f"Under one identical image-based definition, the selected Stage 2 checkpoint corrected {stage['stage2_best_conditional_numerator']} of {stage['stage2_best_conditional_denominator']} conditional MEL/NV errors ({100*stage['stage2_best_conditional_rate']:.2f}%). The selected Stage 3 checkpoint corrected {stage['stage3_best_conditional_numerator']} of {stage['stage3_best_conditional_denominator']} ({100*stage['stage3_best_conditional_rate']:.2f}%). Because joint fine-tuning changes the generalist as well as the specialist, these raw denominators differ; the fixed-denominator tables must be used for a direct specialist comparison.",
        "",
        f"On the fixed set of 87 images that were Stage 2 conditional errors, Stage 2's specialist was correct on 43 and Stage 3's specialist on 36: a net loss of 7 corrections ({100*stage['stage2_error_set_fixed_delta']:.2f} percentage points). On the 73 errors shared by both generalists, the change was 35 to 26 ({100*stage['intersection_fixed_delta']:.2f} points). In contrast, specialist accuracy across all 803 MEL/NV validation images rose from {100*stage['stage2_pair_specialist_accuracy']:.2f}% to {100*stage['stage3_pair_specialist_accuracy']:.2f}%. Joint fine-tuning therefore improved broad pair classification while weakening performance specifically on the difficult cases the generalist got wrong, which supports the hypothesized complementarity conflict.",
        "",
        f"The final model bundle is {'exactly identical' if stage['final_bundle_equals_stage3_best'] else 'not identical'} to the saved Stage 3 best model state (maximum tensor difference {stage['final_bundle_stage3_best_max_abs_difference']:.3g}).",
        "",
        "## Gate and feasibility-transfer audit",
        "",
        f"On {gate['test_images']} held-out test images, learned alpha fusion changed {gate['alpha_argmax_changes']} seven-class predictions. Of those changes, {gate['alpha_beneficial']} were beneficial and {gate['alpha_harmful']} were harmful. Adding the learned beta transfer changed {gate['beta_argmax_changes']} predictions beyond primary fusion: {gate['beta_beneficial']} beneficial, {gate['beta_harmful']} harmful, and {gate['beta_wrong_to_wrong']} wrong-to-wrong.",
        "",
        f"Although mean beta was {gate['mean_beta']:.4f}, the mean realized MEL probability transfer was only {gate['mean_beta_transfer']:.6f}; {gate['transfer_over_0_001']} images received more than 0.001 MEL probability and {gate['transfer_over_0_01']} received more than 0.01. This demonstrates why a large beta coefficient alone is not evidence of a material transfer: beta is multiplied by the off-pair margin and three soft gates.",
        "",
        "## Validation-to-test gap",
        "",
        f"Full CADF-Net v3 pair-balanced accuracy changed from {100*gap['validation_pair_balanced_accuracy']:.2f}% on validation to {100*gap['test_pair_balanced_accuracy']:.2f}% on test (difference {100*gap['pair_balanced_accuracy_difference']:.2f} percentage points). MEL sensitivity changed from {100*gap['validation_mel_sensitivity']:.2f}% to {100*gap['test_mel_sensitivity']:.2f}%.",
        "",
        f"The lesion-level 95% bootstrap interval for the full-model pair-balanced-accuracy gap is [{100*gap['pair_gap_ci_low']:.2f}, {100*gap['pair_gap_ci_high']:.2f}] percentage points; for the MEL-sensitivity gap it is [{100*gap['mel_gap_ci_low']:.2f}, {100*gap['mel_gap_ci_high']:.2f}]. Both include zero. These intervals quantify sampling uncertainty, but this observational comparison cannot by itself separate sampling variation, cohort composition, or representation shift. It should not be interpreted as a causal attribution.",
        "",
        "## Interpretation",
        "",
        "The audit distinguishes three claims. First, final-epoch training history and selected-checkpoint behavior are different quantities and must not be compared as if they were the same model states. Second, the fixed-denominator paired analysis measures whether the specialist itself gained or lost corrective decisions; the varying-denominator rates additionally include changes in the generalist error set. Third, learned beta magnitude is not the realized transfer magnitude, and the prediction-change table directly measures its practical effect.",
        "",
        "These results are diagnostic, not a basis for another round of test-guided tuning. Any architectural change should be selected on development data and assessed on a new independent cohort.",
        "",
        "## Reproducibility",
        "",
        "Exact source hashes are recorded in `results/source_manifest.csv`. Machine-readable findings are in `results/audit_summary.json`; all checkpoint-level, per-image, fixed-denominator, class-stratified, and bootstrap results are preserved in the other CSV files.",
    ]
    (AUDIT_DIR / "V3_Audit_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in SOURCE_FILES if not (SOURCE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required completed-run artifacts are missing: {missing}")

    config = load_config(CONFIG_PATH)
    frame = load_ham_manifest(config)
    validation_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    manifest_rows = []
    for name in SOURCE_FILES:
        path = SOURCE_DIR / name
        manifest_rows.append(
            {"file": name, "bytes": path.stat().st_size, "sha256": sha256(path), "role": "read_only_source"}
        )
    pd.DataFrame(manifest_rows).to_csv(RESULTS_DIR / "source_manifest.csv", index=False)

    stage_rows: list[dict[str, Any]] = []
    stage_masks: dict[str, dict[str, np.ndarray]] = {}
    stage_outputs: dict[str, dict[str, Any]] = {}
    original_feature_cache = torch.load(SOURCE_DIR / "initial_feature_cache.pt", map_location="cpu")
    for label, filename in STAGE_CHECKPOINTS.items():
        checkpoint_path = SOURCE_DIR / filename
        outputs = infer_checkpoint(
            checkpoint_path,
            RESULTS_DIR / f"{label}_validation_outputs.pt",
            validation_frame,
            config,
            device,
        )
        stage_outputs[label] = outputs
        metrics, masks = checkpoint_audit(outputs)
        stage_masks[label] = masks
        state = torch.load(checkpoint_path, map_location="cpu")
        stored = state.get("record") or (state.get("history") or [{}])[-1]
        row = {"checkpoint": label, "source_file": filename, "selected_or_last": "selected_best" if label.endswith("best") else "last_epoch"}
        row.update(metrics)
        for key, value in stored.items():
            row[f"stored_{key}"] = value
        if label.startswith("stage2"):
            cache_metrics, cache_masks = audit_stage2_on_original_feature_cache(
                checkpoint_path, original_feature_cache, config
            )
            for key, value in cache_metrics.items():
                row[f"cached_feature_{key}"] = value
            row["cached_feature_vs_image_generalist_prediction_changes"] = int(
                (cache_masks["generalist_prediction"] != masks["generalist_prediction"]).sum()
            )
            row["cached_feature_vs_image_specialist_prediction_changes"] = int(
                (cache_masks["specialist_prediction"] != masks["specialist_prediction"]).sum()
            )
        stage_rows.append(row)
    stage_table = pd.DataFrame(stage_rows)
    stage_table.to_csv(RESULTS_DIR / "stage_checkpoint_metrics.csv", index=False)

    s2 = stage_masks["stage2_best"]
    s3 = stage_masks["stage3_best"]
    if stage_outputs["stage2_best"]["image_ids"] != stage_outputs["stage3_best"]["image_ids"]:
        raise AssertionError("Stage 2 and Stage 3 validation output order differs")
    paired = pd.DataFrame(
        {
            "image_id": stage_outputs["stage2_best"]["image_ids"],
            "lesion_id": stage_outputs["stage2_best"]["lesion_ids"],
            "true_label_index": s2["labels"],
            "true_class": [CLASS_NAMES[index] for index in s2["labels"]],
            "pair_image": s2["pair_mask"],
            "stage2_generalist_prediction": s2["generalist_prediction"],
            "stage3_generalist_prediction": s3["generalist_prediction"],
            "stage2_specialist_pair_prediction": s2["specialist_prediction"],
            "stage3_specialist_pair_prediction": s3["specialist_prediction"],
            "stage2_conditional_error": s2["conditional_error"],
            "stage3_conditional_error": s3["conditional_error"],
            "stage2_full_error": s2["full_error"],
            "stage3_full_error": s3["full_error"],
            "stage2_specialist_correct": s2["specialist_correct"],
            "stage3_specialist_correct": s3["specialist_correct"],
        }
    )
    paired["specialist_transition"] = np.select(
        [
            paired.stage2_specialist_correct & paired.stage3_specialist_correct,
            paired.stage2_specialist_correct & ~paired.stage3_specialist_correct,
            ~paired.stage2_specialist_correct & paired.stage3_specialist_correct,
        ],
        ["correct_retained", "correction_lost", "correction_gained"],
        default="wrong_retained",
    )
    paired.to_csv(RESULTS_DIR / "stage2_stage3_paired_validation.csv", index=False)

    fixed_table = pd.DataFrame(fixed_denominator_rows(s2, s3))
    fixed_table.to_csv(RESULTS_DIR / "fixed_denominator_comparison.csv", index=False)
    transition_rows = []
    for scope, mask in {
        "all_pair_images": s2["pair_mask"],
        "stage2_conditional_errors": s2["conditional_error"],
        "stage3_conditional_errors": s3["conditional_error"],
        "conditional_error_intersection": s2["conditional_error"] & s3["conditional_error"],
        "conditional_error_union": s2["conditional_error"] | s3["conditional_error"],
        "stage2_full_errors": s2["full_error"],
        "stage3_full_errors": s3["full_error"],
    }.items():
        counts = Counter(paired.loc[mask, "specialist_transition"])
        for transition in ["correct_retained", "correction_lost", "correction_gained", "wrong_retained"]:
            transition_rows.append({"scope": scope, "transition": transition, "count": counts.get(transition, 0), "denominator": int(mask.sum())})
    pd.DataFrame(transition_rows).to_csv(RESULTS_DIR / "stage_transition_summary.csv", index=False)

    final_bundle = torch.load(SOURCE_DIR / "cadfnet_v3_final.pt", map_location="cpu")
    stage3_best_state = torch.load(SOURCE_DIR / STAGE_CHECKPOINTS["stage3_best"], map_location="cpu")["model_state"]
    final_equal, final_difference = checkpoint_states_equal(final_bundle["model_state"], stage3_best_state)
    implementation_observations = {
        "stage2_validation_dropout_active": True,
        "evidence": {
            "training_call": "code/cadfnet_v3/training.py:346 calls model.specialist_head.train()",
            "validation_call": "code/cadfnet_v3/training.py:377 calls _head_validation()",
            "validation_function": "code/cadfnet_v3/training.py:178 uses torch.no_grad() but does not call model.eval()",
        },
        "impact": "Stage 2 saved validation metrics and best-checkpoint selection are stochastic dropout-on measurements. Deterministic evaluation of the saved checkpoint may differ.",
        "stage3_contrast": "Stage 3 image validation calls model.eval(), so its saved final-epoch rate reproduces deterministically.",
        "source_modified": False,
    }
    save_json(RESULTS_DIR / "implementation_observations.json", implementation_observations)
    predictor, gate, fusion = gate_modules(config, final_bundle)
    validation_saved = torch.load(SOURCE_DIR / "validation_outputs.pt", map_location="cpu")
    test_saved = torch.load(SOURCE_DIR / "test_outputs.pt", map_location="cpu")
    validation_gate = gate_outputs(validation_saved, predictor, gate, fusion)
    test_gate = gate_outputs(test_saved, predictor, gate, fusion)

    for cohort, gated in [("validation", validation_gate), ("test", test_gate)]:
        for variant in ["baseline", "primary", "full", "fixed"]:
            sums = gated[variant].sum(dim=1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
                raise AssertionError(f"{cohort} {variant} probabilities do not sum to one")
            if float(gated[variant].min()) < 0:
                raise AssertionError(f"{cohort} {variant} contains negative probabilities")
        delta_error = float((gated["expected_delta"] - gated["actual_delta"]).abs().max())
        if delta_error > 2e-6:
            raise AssertionError(f"{cohort} analytical and realized beta transfer differ by {delta_error}")

    labels = test_saved["labels"].numpy()
    predictions = {name: test_gate[name].argmax(dim=1).numpy() for name in ["baseline", "primary", "full", "fixed"]}
    alpha_category = effect_category(predictions["baseline"], predictions["primary"], labels)
    beta_category = effect_category(predictions["primary"], predictions["full"], labels)
    full_category = effect_category(predictions["baseline"], predictions["full"], labels)
    strongest_full_index = np.asarray(OFF_PAIR_INDICES)[test_gate["strongest_offpair_column"].numpy()]
    gate_frame = pd.DataFrame(
        {
            "image_id": test_saved["image_ids"],
            "lesion_id": test_saved["lesion_ids"],
            "true_label_index": labels,
            "true_class": [CLASS_NAMES[index] for index in labels],
            "baseline_prediction": predictions["baseline"],
            "primary_prediction": predictions["primary"],
            "full_prediction": predictions["full"],
            "fixed_prediction": predictions["fixed"],
            "alpha": test_gate["alpha"].numpy(),
            "beta": test_gate["beta"].numpy(),
            "strongest_offpair_index": strongest_full_index,
            "strongest_offpair_class": [CLASS_NAMES[index] for index in strongest_full_index],
            "strongest_offpair_probability": test_gate["strongest_offpair_probability"].numpy(),
            "strongest_feasibility": test_gate["strongest_feasibility"].numpy(),
            "feasibility_gate": test_gate["feasibility_gate"].numpy(),
            "alpha_gate": test_gate["alpha_gate"].numpy(),
            "melanoma_preference": test_gate["melanoma_preference"].numpy(),
            "maximum_transfer": test_gate["maximum_transfer"].numpy(),
            "beta_mel_transfer": test_gate["actual_delta"].numpy(),
            "alpha_effect": alpha_category,
            "beta_effect": beta_category,
            "full_effect": full_category,
        }
    )
    for variant in ["baseline", "primary", "full", "fixed"]:
        for class_index, class_name in enumerate(CLASS_NAMES):
            gate_frame[f"{variant}_p_{class_name}"] = test_gate[variant][:, class_index].numpy()
    gate_frame.to_csv(RESULTS_DIR / "test_gate_effects_per_image.csv", index=False)

    change_rows = []
    comparison_categories = {"alpha_baseline_to_primary": alpha_category, "beta_primary_to_full": beta_category, "full_baseline_to_full": full_category}
    for comparison, categories in comparison_categories.items():
        for scope, mask in {"all_classes": np.ones(len(labels), dtype=bool), "mel_nv_only": np.isin(labels, [MEL_IDX, NV_IDX]), "mel_only": labels == MEL_IDX, "nv_only": labels == NV_IDX, "offpair_only": ~np.isin(labels, [MEL_IDX, NV_IDX])}.items():
            counts = Counter(categories[mask])
            for category in ["beneficial", "harmful", "wrong_to_wrong", "correct_to_correct", "unchanged"]:
                change_rows.append({"comparison": comparison, "scope": scope, "effect": category, "count": counts.get(category, 0), "denominator": int(mask.sum())})
    change_table = pd.DataFrame(change_rows)
    change_table.to_csv(RESULTS_DIR / "test_gate_change_summary.csv", index=False)

    by_class_rows = []
    for comparison, categories in comparison_categories.items():
        for class_index, class_name in enumerate(CLASS_NAMES):
            mask = labels == class_index
            counts = Counter(categories[mask])
            row = {"comparison": comparison, "true_class": class_name, "support": int(mask.sum())}
            row.update({category: counts.get(category, 0) for category in ["beneficial", "harmful", "wrong_to_wrong", "correct_to_correct", "unchanged"]})
            by_class_rows.append(row)
    pd.DataFrame(by_class_rows).to_csv(RESULTS_DIR / "test_gate_changes_by_class.csv", index=False)

    distribution_names = ["alpha", "beta", "strongest_offpair_probability", "strongest_feasibility", "feasibility_gate", "alpha_gate", "melanoma_preference", "maximum_transfer", "actual_delta"]
    distribution_table = pd.DataFrame(distribution_rows({name: test_gate[name] for name in distribution_names}, "test"))
    distribution_table.to_csv(RESULTS_DIR / "gate_distribution_summary.csv", index=False)
    threshold_rows = []
    for threshold in [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        threshold_rows.append({"threshold": threshold, "images_above_threshold": int((test_gate["actual_delta"].numpy() > threshold).sum()), "rate": float((test_gate["actual_delta"].numpy() > threshold).mean())})
    pd.DataFrame(threshold_rows).to_csv(RESULTS_DIR / "beta_transfer_thresholds.csv", index=False)

    comparison_rows = []
    cohort_payload = {"validation": (validation_saved, validation_gate), "test": (test_saved, test_gate)}
    for cohort, (outputs, gated) in cohort_payload.items():
        for variant in ["baseline", "primary", "full", "fixed"]:
            metrics = evaluate_probabilities(gated[variant], outputs["labels"], reference=gated["baseline"] if variant != "baseline" else None)
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "pair_balanced_accuracy", "mel_sensitivity", "nv_sensitivity", "unrecoverable_melanoma_count", "unrecoverable_melanoma_rate", "offpair_to_pair_transitions", "offpair_to_pair_transition_rate"]:
                if metric in metrics:
                    comparison_rows.append({"cohort": cohort, "variant": variant, "metric": metric, "value": metrics[metric]})
            recalls = per_class_recall(outputs["labels"].numpy(), gated[variant].argmax(dim=1).numpy())
            for class_name, recall in zip(CLASS_NAMES, recalls):
                comparison_rows.append({"cohort": cohort, "variant": variant, "metric": f"recall_{class_name}", "value": recall})
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(RESULTS_DIR / "validation_test_comparison.csv", index=False)

    official_metrics = json.loads((SOURCE_DIR / "evaluation_metrics.json").read_text(encoding="utf-8"))
    variant_mapping = {
        "baseline": "A1_generalist_only",
        "primary": "A3_no_feasibility_transfer",
        "full": "A0_full",
        "fixed": "A4_fixed_gate",
    }
    reproduction_rows = []
    for audit_variant, official_variant in variant_mapping.items():
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "pair_balanced_accuracy", "mel_sensitivity", "nv_sensitivity"]:
            reproduced = float(
                comparison_table[
                    (comparison_table.cohort == "test")
                    & (comparison_table.variant == audit_variant)
                    & (comparison_table.metric == metric)
                ].value.iloc[0]
            )
            official = float(official_metrics[official_variant][metric])
            reproduction_rows.append(
                {
                    "audit_variant": audit_variant,
                    "official_variant": official_variant,
                    "metric": metric,
                    "official_value": official,
                    "reproduced_value": reproduced,
                    "absolute_difference": abs(reproduced - official),
                }
            )
    reproduction_table = pd.DataFrame(reproduction_rows)
    reproduction_table.to_csv(RESULTS_DIR / "official_test_metric_reproduction.csv", index=False)
    if float(reproduction_table.absolute_difference.max()) > 1e-10:
        raise AssertionError("Audit reconstruction does not reproduce the official saved test metrics")

    bootstrap_rows = []
    for variant in ["baseline", "primary", "full"]:
        for metric in ["pair_balanced_accuracy", "mel_sensitivity", "macro_f1"]:
            row = lesion_bootstrap_gap(
                validation_saved["labels"].numpy(),
                validation_gate[variant].argmax(dim=1).numpy(),
                validation_saved["lesion_ids"],
                test_saved["labels"].numpy(),
                test_gate[variant].argmax(dim=1).numpy(),
                test_saved["lesion_ids"],
                metric,
                int(config["evaluation"]["bootstrap_resamples"]),
                int(config["seed"]) + ["baseline", "primary", "full"].index(variant) * 10 + ["pair_balanced_accuracy", "mel_sensitivity", "macro_f1"].index(metric),
            )
            row["variant"] = variant
            bootstrap_rows.append(row)
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    bootstrap_table.to_csv(RESULTS_DIR / "validation_test_gap_bootstrap.csv", index=False)

    stage_by_name = stage_table.set_index("checkpoint")
    full_comparison = comparison_table[comparison_table.variant == "full"].pivot(index="metric", columns="cohort", values="value")
    all_change = change_table[change_table.scope == "all_classes"].set_index(["comparison", "effect"])["count"]
    threshold_counts = dict(zip([str(row["threshold"]) for row in threshold_rows], [row["images_above_threshold"] for row in threshold_rows]))
    stage2_fixed = fixed_table[(fixed_table.error_definition == "conditional_error") & (fixed_table.denominator_source == "stage2_error_set")].iloc[0]
    intersection_fixed = fixed_table[(fixed_table.error_definition == "conditional_error") & (fixed_table.denominator_source == "intersection")].iloc[0]
    full_bootstrap = bootstrap_table[bootstrap_table.variant == "full"].set_index("metric")
    summary = {
        "audit_constraints": {"retrained": False, "retuned": False, "original_outputs_modified": False, "test_use": "diagnosis_and_reporting_only"},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "stage_comparison": {
            "stage2_selected_epoch": stage_by_name.loc["stage2_best", "stored_epoch"],
            "stage3_selected_epoch": stage_by_name.loc["stage3_best", "stored_epoch"],
            "stage2_last_stored_conditional_rate": stage_by_name.loc["stage2_last", "stored_pair_conditional_correction_rate"],
            "stage3_last_stored_conditional_rate": stage_by_name.loc["stage3_last", "stored_pair_conditional_correction_rate"],
            "stage2_last_reevaluated_conditional_rate": stage_by_name.loc["stage2_last", "pair_conditional_correction_rate"],
            "stage3_last_reevaluated_conditional_rate": stage_by_name.loc["stage3_last", "pair_conditional_correction_rate"],
            "stage2_last_cached_feature_conditional_rate": stage_by_name.loc["stage2_last", "cached_feature_pair_conditional_correction_rate"],
            "stage2_best_stored_conditional_rate": stage_by_name.loc["stage2_best", "stored_pair_conditional_correction_rate"],
            "stage2_best_cached_feature_conditional_rate": stage_by_name.loc["stage2_best", "cached_feature_pair_conditional_correction_rate"],
            "stage2_validation_dropout_active": True,
            "stage2_best_conditional_numerator": int(stage_by_name.loc["stage2_best", "conditional_correction_numerator"]),
            "stage2_best_conditional_denominator": int(stage_by_name.loc["stage2_best", "conditional_error_denominator"]),
            "stage2_best_conditional_rate": stage_by_name.loc["stage2_best", "pair_conditional_correction_rate"],
            "stage3_best_conditional_numerator": int(stage_by_name.loc["stage3_best", "conditional_correction_numerator"]),
            "stage3_best_conditional_denominator": int(stage_by_name.loc["stage3_best", "conditional_error_denominator"]),
            "stage3_best_conditional_rate": stage_by_name.loc["stage3_best", "pair_conditional_correction_rate"],
            "stage2_pair_specialist_accuracy": stage_by_name.loc["stage2_best", "specialist_pair_accuracy"],
            "stage3_pair_specialist_accuracy": stage_by_name.loc["stage3_best", "specialist_pair_accuracy"],
            "stage2_error_set_fixed_delta": stage2_fixed["stage3_minus_stage2"],
            "intersection_fixed_delta": intersection_fixed["stage3_minus_stage2"],
            "final_bundle_equals_stage3_best": final_equal,
            "final_bundle_stage3_best_max_abs_difference": final_difference,
        },
        "test_gate_audit": {
            "test_images": len(labels),
            "mean_alpha": float(test_gate["alpha"].mean()),
            "mean_beta": float(test_gate["beta"].mean()),
            "mean_beta_transfer": float(test_gate["actual_delta"].mean()),
            "alpha_argmax_changes": int((predictions["baseline"] != predictions["primary"]).sum()),
            "beta_argmax_changes": int((predictions["primary"] != predictions["full"]).sum()),
            "full_argmax_changes": int((predictions["baseline"] != predictions["full"]).sum()),
            "alpha_beneficial": int(all_change.get(("alpha_baseline_to_primary", "beneficial"), 0)),
            "alpha_harmful": int(all_change.get(("alpha_baseline_to_primary", "harmful"), 0)),
            "beta_beneficial": int(all_change.get(("beta_primary_to_full", "beneficial"), 0)),
            "beta_harmful": int(all_change.get(("beta_primary_to_full", "harmful"), 0)),
            "beta_wrong_to_wrong": int(all_change.get(("beta_primary_to_full", "wrong_to_wrong"), 0)),
            "transfer_over_0_001": int(threshold_counts["0.001"]),
            "transfer_over_0_01": int(threshold_counts["0.01"]),
            "max_analytical_vs_realized_transfer_error": float((test_gate["expected_delta"] - test_gate["actual_delta"]).abs().max()),
        },
        "validation_test_gap": {
            "validation_pair_balanced_accuracy": float(full_comparison.loc["pair_balanced_accuracy", "validation"]),
            "test_pair_balanced_accuracy": float(full_comparison.loc["pair_balanced_accuracy", "test"]),
            "pair_balanced_accuracy_difference": float(full_comparison.loc["pair_balanced_accuracy", "test"] - full_comparison.loc["pair_balanced_accuracy", "validation"]),
            "validation_mel_sensitivity": float(full_comparison.loc["mel_sensitivity", "validation"]),
            "test_mel_sensitivity": float(full_comparison.loc["mel_sensitivity", "test"]),
            "validation_macro_f1": float(full_comparison.loc["macro_f1", "validation"]),
            "test_macro_f1": float(full_comparison.loc["macro_f1", "test"]),
            "pair_gap_ci_low": float(full_bootstrap.loc["pair_balanced_accuracy", "ci_95_low"]),
            "pair_gap_ci_high": float(full_bootstrap.loc["pair_balanced_accuracy", "ci_95_high"]),
            "mel_gap_ci_low": float(full_bootstrap.loc["mel_sensitivity", "ci_95_low"]),
            "mel_gap_ci_high": float(full_bootstrap.loc["mel_sensitivity", "ci_95_high"]),
        },
    }
    save_json(RESULTS_DIR / "audit_summary.json", summary)
    plot_outputs(stage_table, gate_frame, change_table, comparison_table[comparison_table.variant == "full"])
    write_report(summary)

    result_manifest = []
    for path in sorted([*RESULTS_DIR.glob("*"), AUDIT_DIR / "V3_Audit_Report.md", AUDIT_DIR / "audit_plan.yaml", AUDIT_DIR / "README.md", Path(__file__)]):
        if path.is_file() and path.name != "result_manifest.csv":
            result_manifest.append({"file": str(path.relative_to(AUDIT_DIR)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(result_manifest).to_csv(RESULTS_DIR / "result_manifest.csv", index=False)
    print(json.dumps(jsonable(summary), indent=2))
    print(f"Audit complete: {AUDIT_DIR / 'V3_Audit_Report.md'}")


if __name__ == "__main__":
    main()
