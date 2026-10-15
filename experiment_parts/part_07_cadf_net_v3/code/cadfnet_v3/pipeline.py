from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .checkpoint import atomic_torch_save
from .config import load_config
from .data import load_bcn_manifest, load_ham_manifest, make_loader, stratified_smoke_sample
from .fusion import build_gate_inputs
from .metrics import paired_lesion_bootstrap
from .model import CADFNet
from .training import (
    cache_initial_features,
    collect_outputs,
    evaluate_ablation_variants,
    initialize_components,
    resolve_device,
    set_seed,
    train_stage1_generalist,
    train_stage2_specialist,
    train_stage3_finetune,
    train_stage4_feasibility,
    train_stage5_gate,
    write_histories,
)


def prepare_run(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, torch.device]:
    set_seed(int(config["seed"]))
    frame = load_ham_manifest(config)
    if config.get("smoke", {}).get("enabled"):
        frame = stratified_smoke_sample(
            frame, int(config["smoke"]["max_images"]), int(config["seed"])
        )
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config)
    summary = pd.crosstab(frame["split"], frame["label"], margins=True)
    summary.to_csv(output_dir / "split_summary.csv")
    (output_dir / "run_environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "images": int(len(frame)),
                "lesions": int(frame["lesion_id"].nunique()),
                "hourly_checkpoint_seconds": float(
                    config["training"]["checkpoint_interval_seconds"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame, output_dir, device


def preflight_check(
    model: CADFNet,
    frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    sample_frame = frame.iloc[: min(2, len(frame))].reset_index(drop=True)
    batch = next(
        iter(
            make_loader(
                sample_frame,
                config,
                training=False,
                batch_size=len(sample_frame),
            )
        )
    )
    model.to(device).eval()
    with torch.no_grad():
        generalist_logits, specialist_logits = model.forward_views(
            batch["generalist_view"].to(device), batch["specialist_view"].to(device)
        )
    checks = {
        "generalist_view_shape": list(batch["generalist_view"].shape),
        "specialist_view_shape": list(batch["specialist_view"].shape),
        "color_stats_shape": list(batch["color_stats"].shape),
        "generalist_logits_shape": list(generalist_logits.shape),
        "specialist_logits_shape": list(specialist_logits.shape),
        "finite": bool(
            torch.isfinite(generalist_logits).all() and torch.isfinite(specialist_logits).all()
        ),
        "lesion_disjoint": bool(frame.groupby("lesion_id")["split"].nunique().max() == 1),
    }
    if checks["generalist_view_shape"][1] != 5 or checks["specialist_view_shape"][1] != 5:
        raise AssertionError("Both shared-encoder inputs must contain five channels")
    if not checks["finite"] or not checks["lesion_disjoint"]:
        raise AssertionError(f"Preflight failed: {checks}")
    return checks


def export_evaluation(
    output_dir: Path,
    outputs: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    variants: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
) -> None:
    serializable_metrics = {name: values for name, values in metrics.items()}
    if "A0_full" in variants:
        serializable_metrics["A0_vs_A1_lesion_bootstrap"] = paired_lesion_bootstrap(
            variants["A0_full"]["fused"],
            F.softmax(outputs["generalist_logits"], dim=1),
            outputs["labels"],
            outputs["lesion_ids"],
            int(config["evaluation"]["bootstrap_resamples"]),
            int(config["seed"]),
        )
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    frame = pd.DataFrame(
        {
            "image_id": outputs["image_ids"],
            "lesion_id": outputs["lesion_ids"],
            "label": outputs["labels"].numpy(),
        }
    )
    baseline = F.softmax(outputs["generalist_logits"], dim=1)
    for class_index in range(7):
        frame[f"A1_prob_{class_index}"] = baseline[:, class_index].numpy()
    for name, values in variants.items():
        for class_index in range(7):
            frame[f"{name}_prob_{class_index}"] = values["fused"][:, class_index].numpy()
        frame[f"{name}_alpha"] = values["alpha"].numpy()
        frame[f"{name}_beta"] = values["beta"].numpy()
    frame.to_csv(output_dir / "test_predictions.csv", index=False)


def run_core_experiment(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frame, output_dir, device = prepare_run(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, predictor, gate, fusion = initialize_components(config)
    preflight = preflight_check(model, frame, config, device)
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )

    head_frame = frame.loc[frame["split"].isin(["train", "val"])].reset_index(drop=True)
    cache = cache_initial_features(
        model, head_frame, output_dir / "initial_feature_cache.pt", config, device
    )
    histories: dict[str, list[dict[str, float]]] = {}
    histories["stage1"] = train_stage1_generalist(model, cache, output_dir, config, device)
    histories["stage2"] = train_stage2_specialist(model, cache, output_dir, config, device)
    histories["stage3"] = train_stage3_finetune(
        model,
        frame.loc[frame["split"] == "train"].reset_index(drop=True),
        frame.loc[frame["split"] == "val"].reset_index(drop=True),
        output_dir,
        config,
        device,
    )
    fusion_outputs = collect_outputs(
        model,
        frame.loc[frame["split"] == "fusion"].reset_index(drop=True),
        output_dir / "fusion_outputs.pt",
        config,
        device,
    )
    validation_outputs = collect_outputs(
        model,
        frame.loc[frame["split"] == "val"].reset_index(drop=True),
        output_dir / "validation_outputs.pt",
        config,
        device,
    )
    histories["stage4"] = train_stage4_feasibility(
        predictor, fusion_outputs, validation_outputs, output_dir, config, device
    )
    histories["stage5"] = train_stage5_gate(
        predictor, gate, fusion, fusion_outputs, validation_outputs, output_dir, config, device
    )
    write_histories(output_dir, histories)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "feasibility_state": predictor.state_dict(),
            "gate_state": gate.state_dict(),
            "config": config,
        },
        output_dir / "cadfnet_v3_final.pt",
    )
    test_outputs = collect_outputs(
        model,
        frame.loc[frame["split"] == "test"].reset_index(drop=True),
        output_dir / "test_outputs.pt",
        config,
        device,
    )
    metrics, variants = evaluate_ablation_variants(
        test_outputs, predictor, gate, fusion, config, device
    )
    export_evaluation(output_dir, test_outputs, metrics, variants, config)
    runtime = {
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "status": "complete",
        "note": "A2 and A5 require retraining and are run by run_trainable_ablation.",
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return {"metrics": metrics, "runtime": runtime, "output_dir": str(output_dir)}


def run_trainable_ablation(config: dict[str, Any], ablation: str) -> dict[str, Any]:
    if ablation not in {"A2", "A5"}:
        raise ValueError("Only A2 and A5 require retraining; use evaluation outputs for A1/A3/A4")
    frame, base_output, device = prepare_run(config)
    ablation_output = base_output / "ablations" / ablation
    ablation_output.mkdir(parents=True, exist_ok=True)
    model, predictor, gate, fusion = initialize_components(config)
    stage1_best = base_output / "stage1_generalist_best.pt"
    if not stage1_best.is_file():
        raise FileNotFoundError("Run the core experiment through Stage 1 before trainable ablations")
    model.load_state_dict(torch.load(stage1_best, map_location="cpu")["model_state"])
    same_view = ablation == "A5"
    standard_ce = ablation == "A2"
    head_frame = frame.loc[frame["split"].isin(["train", "val"])].reset_index(drop=True)
    cache = cache_initial_features(
        model,
        head_frame,
        ablation_output / "initial_feature_cache.pt",
        config,
        device,
        same_view=same_view,
    )
    histories = {
        "stage2": train_stage2_specialist(
            model,
            cache,
            ablation_output,
            config,
            device,
            standard_cross_entropy=standard_ce,
        )
    }
    histories["stage3"] = train_stage3_finetune(
        model,
        frame.loc[frame["split"] == "train"].reset_index(drop=True),
        frame.loc[frame["split"] == "val"].reset_index(drop=True),
        ablation_output,
        config,
        device,
        standard_cross_entropy=standard_ce,
        same_view=same_view,
    )
    fusion_outputs = collect_outputs(
        model,
        frame.loc[frame["split"] == "fusion"].reset_index(drop=True),
        ablation_output / "fusion_outputs.pt",
        config,
        device,
        same_view=same_view,
    )
    validation_outputs = collect_outputs(
        model,
        frame.loc[frame["split"] == "val"].reset_index(drop=True),
        ablation_output / "validation_outputs.pt",
        config,
        device,
        same_view=same_view,
    )
    histories["stage4"] = train_stage4_feasibility(
        predictor, fusion_outputs, validation_outputs, ablation_output, config, device
    )
    histories["stage5"] = train_stage5_gate(
        predictor, gate, fusion, fusion_outputs, validation_outputs, ablation_output, config, device
    )
    write_histories(ablation_output, histories)
    test_outputs = collect_outputs(
        model,
        frame.loc[frame["split"] == "test"].reset_index(drop=True),
        ablation_output / "test_outputs.pt",
        config,
        device,
        same_view=same_view,
    )
    metrics, variants = evaluate_ablation_variants(
        test_outputs, predictor, gate, fusion, config, device
    )
    metrics[f"{ablation}_full"] = metrics.pop("A0_full")
    variants[f"{ablation}_full"] = variants.pop("A0_full")
    export_evaluation(ablation_output, test_outputs, metrics, variants, config)
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "feasibility_state": predictor.state_dict(),
            "gate_state": gate.state_dict(),
            "ablation": ablation,
        },
        ablation_output / f"cadfnet_v3_{ablation}.pt",
    )
    return {"ablation": ablation, "metrics": metrics, "output_dir": str(ablation_output)}


def run_external_evaluation(
    config: dict[str, Any], max_images: int | None = None
) -> dict[str, Any]:
    """Evaluate the already-fixed model on BCN20000 without fitting any component."""
    _, base_output, device = prepare_run(config)
    checkpoint_path = base_output / "cadfnet_v3_final.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Complete the core experiment before external evaluation")
    model, predictor, gate, fusion = initialize_components(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    predictor.load_state_dict(checkpoint["feasibility_state"])
    gate.load_state_dict(checkpoint["gate_state"])
    model.to(device).eval()
    predictor.to(device).eval()
    gate.to(device).eval()
    primary, excluded = load_bcn_manifest(config)
    if max_images is not None:
        primary = primary.iloc[: int(max_images)].reset_index(drop=True)
    external_dir = base_output / ("bcn_external" if max_images is None else f"bcn_external_smoke_{max_images}")
    external_dir.mkdir(parents=True, exist_ok=True)
    outputs = collect_outputs(
        model,
        primary,
        external_dir / "bcn_outputs.pt",
        config,
        device,
    )
    metrics, variants = evaluate_ablation_variants(
        outputs, predictor, gate, fusion, config, device
    )
    export_evaluation(external_dir, outputs, metrics, variants, config)
    status = {
        "included_images": int(len(primary)),
        "included_lesions": int(primary["lesion_id"].nunique()),
        "excluded_images": int(len(excluded)),
        "excluded_diagnoses": excluded["diagnosis"].value_counts(dropna=False).to_dict(),
        "no_external_fitting": True,
    }
    (external_dir / "external_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return {"metrics": metrics, "status": status, "output_dir": str(external_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CADF-Net v3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override")
    parser.add_argument("--trainable-ablation", choices=["A2", "A5"])
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--max-external-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    if args.external:
        result = run_external_evaluation(config, args.max_external_images)
    elif args.trainable_ablation:
        result = run_trainable_ablation(config, args.trainable_ablation)
    else:
        result = run_core_experiment(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
