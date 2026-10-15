from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .checkpoint import atomic_torch_save
from .config import load_config, resolve_device, seed_everything
from .data import MEL_IDX, NV_IDX, build_manifest, create_split_manifest, split_summary, stratified_smoke_sample
from .evaluation import evaluate_probabilities, paired_group_bootstrap
from .fusion import apply_gate, fit_temperature, pair_restricted_fusion, train_gate
from .model import HardPairNet
from .training import extract_feature_cache, initialize_heads, predict_model, train_image_model


def _calibrated_probabilities(
    predictions: dict[str, Any], generalist_temperature: float, specialist_temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.softmax(predictions["generalist_logits"].float() / generalist_temperature, 1)
    s = torch.softmax(predictions["specialist_logits"].float() / specialist_temperature, 1)
    return p, s


def _fit_temperatures(predictions: dict[str, Any], device: torch.device) -> tuple[float, float]:
    logits_7 = predictions["generalist_logits"].to(device)
    labels = predictions["labels"].to(device)
    generalist_temperature = fit_temperature(logits_7, labels)
    pair_mask = (labels == MEL_IDX) | (labels == NV_IDX)
    binary_targets = (labels[pair_mask] == NV_IDX).long()
    specialist_temperature = fit_temperature(
        predictions["specialist_logits"].to(device)[pair_mask], binary_targets
    )
    return generalist_temperature, specialist_temperature


def run_pipeline(
    config_path: str | Path = "config.yaml",
    smoke: bool = False,
    max_images: int = 100,
) -> dict[str, Any]:
    config = load_config(config_path)
    config["_smoke"] = smoke
    if smoke:
        config["dataset"]["num_workers"] = 0
        config["training"]["checkpoint_interval_seconds"] = min(
            10.0, float(config["training"]["checkpoint_interval_seconds"])
        )
        config["fusion"]["gate_max_steps"] = min(50, int(config["fusion"]["gate_max_steps"]))
        config["fusion"]["gate_validation_interval"] = min(
            10, int(config["fusion"]["gate_validation_interval"])
        )
        config["evaluation"]["bootstrap_resamples"] = min(
            25, int(config["evaluation"]["bootstrap_resamples"])
        )
    seed_everything(int(config["seed"]))
    device = resolve_device(config)
    full_run_name = config["paths"].get("full_run_dir", "full")
    run_dir = Path(config["paths"]["output_dir"]) / ("smoke" if smoke else full_run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(config, verify_images=False)
    full_split_path = Path(config["paths"]["output_dir"]) / "ham10000_splits.csv"
    if full_split_path.exists():
        split_manifest = pd.read_csv(full_split_path)
        expected = manifest.image_id.tolist()
        if split_manifest.image_id.tolist() != expected:
            split_manifest = create_split_manifest(manifest, config, full_split_path)
    else:
        split_manifest = create_split_manifest(manifest, config, full_split_path)
    if smoke:
        split_manifest = stratified_smoke_sample(split_manifest, max_images, int(config["seed"]))
        split_manifest.to_csv(run_dir / "smoke_manifest.csv", index=False)
    split_summary(split_manifest).to_csv(run_dir / "split_summary.csv")

    print(f"Running on {device}; {len(split_manifest)} images; outputs: {run_dir}")
    pretrained = HardPairNet(pretrained=True)
    cache = extract_feature_cache(pretrained, split_manifest, run_dir / "pretrained_features.pt", config, device)
    initialization = initialize_heads(
        pretrained, cache, run_dir / "head_initialization.pt", config, device
    )

    initial_state = initialization["model_state"]
    train_frame = split_manifest.loc[split_manifest.split == "train"].reset_index(drop=True)
    val_frame = split_manifest.loc[split_manifest.split == "val"].reset_index(drop=True)

    baseline = HardPairNet(pretrained=True)
    baseline.load_state_dict(initial_state)
    baseline, baseline_training = train_image_model(
        baseline, train_frame, val_frame, "baseline", run_dir, config, device, multitask=False
    )
    multitask = HardPairNet(pretrained=True)
    multitask.load_state_dict(initial_state)
    multitask, multitask_training = train_image_model(
        multitask, train_frame, val_frame, "multitask", run_dir, config, device, multitask=True
    )

    frames = {
        split: split_manifest.loc[split_manifest.split == split].reset_index(drop=True)
        for split in ("fusion", "val", "test")
    }
    predictions: dict[str, dict[str, dict[str, Any]]] = {"baseline": {}, "multitask": {}}
    for model_name, model in (("baseline", baseline), ("multitask", multitask)):
        for split, frame in frames.items():
            predictions[model_name][split] = predict_model(model, frame, config, device)
    atomic_torch_save(predictions, run_dir / "cached_predictions.pt")

    base_t7, base_t2 = _fit_temperatures(predictions["baseline"]["fusion"], device)
    multi_t7, multi_t2 = _fit_temperatures(predictions["multitask"]["fusion"], device)
    calibrated: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {
        "baseline": {}, "multitask": {}
    }
    for split in frames:
        calibrated["baseline"][split] = _calibrated_probabilities(
            predictions["baseline"][split], base_t7, base_t2
        )
        calibrated["multitask"][split] = _calibrated_probabilities(
            predictions["multitask"][split], multi_t7, multi_t2
        )

    fit_p, fit_s = calibrated["multitask"]["fusion"]
    val_p, val_s = calibrated["multitask"]["val"]
    gate, gate_stats = train_gate(
        fit_p,
        fit_s,
        predictions["multitask"]["fusion"]["labels"],
        val_p,
        val_s,
        predictions["multitask"]["val"]["labels"],
        config,
        device,
    )
    atomic_torch_save(
        {
            "gate_state": gate.state_dict(),
            "mean": gate_stats["mean"],
            "std": gate_stats["std"],
            "validation_loss": gate_stats["validation_loss"],
        },
        run_dir / "learned_gate.pt",
    )

    test_labels = predictions["multitask"]["test"]["labels"]
    baseline_test_p, _ = calibrated["baseline"]["test"]
    multitask_test_p, multitask_test_s = calibrated["multitask"]["test"]
    fixed_test = pair_restricted_fusion(
        multitask_test_p, multitask_test_s, float(config["fusion"]["fixed_alpha"])
    )
    learned_test, test_alpha = apply_gate(gate, gate_stats, multitask_test_p, multitask_test_s)

    metrics = {
        "baseline": evaluate_probabilities(baseline_test_p, test_labels),
        "auxiliary_only": evaluate_probabilities(multitask_test_p, test_labels),
        "fixed_fusion": evaluate_probabilities(fixed_test, test_labels, multitask_test_p),
        "learned_fusion": evaluate_probabilities(learned_test, test_labels, multitask_test_p),
    }
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    metrics["learned_vs_auxiliary_bootstrap"] = paired_group_bootstrap(
        learned_test,
        multitask_test_p,
        test_labels,
        predictions["multitask"]["test"]["lesion_ids"],
        resamples,
        int(config["seed"]),
    )
    metrics["fixed_vs_auxiliary_bootstrap"] = paired_group_bootstrap(
        fixed_test,
        multitask_test_p,
        test_labels,
        predictions["multitask"]["test"]["lesion_ids"],
        resamples,
        int(config["seed"]) + 1,
    )
    metrics["run"] = {
        "smoke": smoke,
        "image_count": len(split_manifest),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "generalist_temperatures": {"baseline": base_t7, "multitask": multi_t7},
        "specialist_temperature": multi_t2,
        "mean_learned_alpha": float(test_alpha.mean()),
        "baseline_epochs": len(baseline_training["history"]),
        "multitask_epochs": len(multitask_training["history"]),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    atomic_torch_save(
        {
            "labels": test_labels,
            "baseline": baseline_test_p,
            "auxiliary_only": multitask_test_p,
            "fixed_fusion": fixed_test,
            "learned_fusion": learned_test,
            "alpha": test_alpha,
            "image_ids": predictions["multitask"]["test"]["image_ids"],
            "lesion_ids": predictions["multitask"]["test"]["lesion_ids"],
        },
        run_dir / "test_probabilities.pt",
    )
    print(json.dumps(metrics["run"], indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HardPair-Net-Lite experiments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Exercise all stages on a small sample")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()
    run_pipeline(args.config, smoke=args.smoke, max_images=args.max_images)


if __name__ == "__main__":
    main()

