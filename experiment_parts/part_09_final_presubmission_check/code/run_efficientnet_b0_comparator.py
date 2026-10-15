from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


SCRIPT = Path(__file__).resolve()
PART = SCRIPT.parent.parent
ROOT = PART.parent.parent
CORE_CODE = ROOT / "experiment_parts" / "part_01_ham10000_core_experiment" / "code"
sys.path.insert(0, str(CORE_CODE))

from hardpair.checkpoint import (  # noqa: E402
    CheckpointManager,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)
from hardpair.config import seed_everything  # noqa: E402
from hardpair.data import (  # noqa: E402
    CLASS_NAMES,
    HAM10000Dataset,
    MEL_IDX,
    ResumableRandomSampler,
    build_transform,
    seed_worker,
    stratified_smoke_sample,
)
from hardpair.evaluation import evaluate_probabilities, paired_group_bootstrap  # noqa: E402
from hardpair.fusion import fit_temperature  # noqa: E402
from hardpair.training import class_weights  # noqa: E402


CONFIG = {
    "experiment": "EfficientNet-B0 same-split comparator",
    "seed": 42,
    "image_size": 192,
    "batch_size": 16,
    "num_workers": 2,
    "epochs": 20,
    "backbone_lr": 2e-5,
    "classifier_lr": 5e-4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "unfreeze_last_feature_modules": 3,
    "checkpoint_interval_seconds": 3600,
    "bootstrap_resamples": 2000,
    "selection_key": "validation macro-F1, with melanoma sensitivity as tie-breaker",
    "calibration_split": "fusion",
    "test_use": "one evaluation after training, checkpoint selection, and calibration were frozen",
}


class EfficientNetComparator(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        input_features = model.classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(input_features, 7))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.features(images)
        values = self.avgpool(values)
        return self.classifier(torch.flatten(values, 1))

    def freeze_for_partial_finetuning(self, last_modules: int) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for module in self.features[-last_modules:]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def freeze_batchnorm_statistics(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False


def loader_kwargs(device: torch.device, workers: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if workers:
        result["persistent_workers"] = True
    return result


@torch.inference_mode()
def predict(
    model: nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    workers: int,
) -> dict[str, Any]:
    dataset = HAM10000Dataset(frame, build_transform(CONFIG["image_size"], training=False))
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        **loader_kwargs(device, workers),
    )
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    image_ids: list[str] = []
    lesion_ids: list[str] = []
    for batch in loader:
        logits.append(model(batch["image"].to(device, non_blocking=True)).cpu())
        labels.append(batch["label"].long().cpu())
        image_ids.extend(batch["image_id"])
        lesion_ids.extend(batch["lesion_id"])
    return {
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "image_ids": image_ids,
        "lesion_ids": lesion_ids,
    }


def validation_score(model: nn.Module, frame: pd.DataFrame, device: torch.device, workers: int) -> tuple[float, float]:
    output = predict(model, frame, device, workers)
    truth = output["labels"].numpy()
    predicted = output["logits"].argmax(1).numpy()
    macro_f1 = float(f1_score(truth, predicted, labels=range(7), average="macro", zero_division=0))
    mel_mask = truth == MEL_IDX
    mel_sensitivity = float((predicted[mel_mask] == MEL_IDX).mean())
    return macro_f1, mel_sensitivity


def train(
    model: EfficientNetComparator,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    smoke: bool,
) -> tuple[EfficientNetComparator, list[dict[str, float]]]:
    latest = output_dir / "efficientnet_b0_latest.pt"
    best = output_dir / "efficientnet_b0_best.pt"
    interval = 10.0 if smoke else CONFIG["checkpoint_interval_seconds"]
    manager = CheckpointManager(latest, interval)
    epochs = 1 if smoke else CONFIG["epochs"]
    workers = 0 if smoke else CONFIG["num_workers"]
    if not manager.exists:
        seed_everything(CONFIG["seed"])

    model.freeze_for_partial_finetuning(CONFIG["unfreeze_last_feature_modules"])
    backbone_parameters = [p for p in model.features.parameters() if p.requires_grad]
    classifier_parameters = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": CONFIG["backbone_lr"]},
            {"params": classifier_parameters, "lr": CONFIG["classifier_lr"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(torch.tensor(train_frame["label"].to_numpy())).to(device)
    )
    dataset = HAM10000Dataset(train_frame, build_transform(CONFIG["image_size"], training=True))
    model.to(device)
    epoch = seen = 0
    loss_sum = 0.0
    loss_count = 0
    best_key = (-float("inf"), -float("inf"))
    history: list[dict[str, float]] = []
    started = time.monotonic()

    if manager.exists:
        state = manager.load(device)
        model.load_state_dict(state["model_state"])
        if state.get("complete") and best.is_file():
            model.load_state_dict(torch.load(best, map_location=device)["model_state"])
            return model, state["history"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        epoch, seen = int(state["epoch"]), int(state["seen"])
        loss_sum, loss_count = float(state["loss_sum"]), int(state["loss_count"])
        best_key = tuple(state["best_key"])
        history = state["history"]
        restore_rng_state(state.get("rng_state"))

    def payload(complete: bool) -> dict[str, Any]:
        return {
            "complete": complete,
            "epoch": epoch,
            "seen": seen,
            "loss_sum": loss_sum,
            "loss_count": loss_count,
            "best_key": best_key,
            "history": history,
            "elapsed_seconds_current_session": time.monotonic() - started,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "config": CONFIG,
        }

    while epoch < epochs:
        sampler = ResumableRandomSampler(dataset, CONFIG["seed"], epoch, seen)
        loader = DataLoader(
            dataset,
            batch_size=CONFIG["batch_size"],
            sampler=sampler,
            **loader_kwargs(device, workers),
        )
        model.train()
        model.freeze_batchnorm_statistics()
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                backbone_parameters + classifier_parameters,
                CONFIG["gradient_clip_norm"],
            )
            optimizer.step()
            count = len(labels)
            seen += count
            loss_sum += float(loss.detach()) * count
            loss_count += count
            manager.maybe_save(lambda: payload(False))

        score = validation_score(model, val_frame, device, workers)
        record = {
            "epoch": epoch + 1,
            "training_loss": loss_sum / max(1, loss_count),
            "validation_macro_f1": score[0],
            "validation_mel_sensitivity": score[1],
        }
        history.append(record)
        if score > best_key:
            best_key = score
            atomic_torch_save(
                {
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "selection_key": score,
                    "epoch": epoch + 1,
                    "config": CONFIG,
                },
                best,
            )
        scheduler.step()
        epoch += 1
        seen, loss_sum, loss_count = 0, 0.0, 0
        manager.save(payload(False))
        print(json.dumps(record), flush=True)

    manager.save(payload(True))
    model.load_state_dict(torch.load(best, map_location=device)["model_state"])
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return model, history


def evaluate_once(
    model: EfficientNetComparator,
    manifest: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    metrics_path = output_dir / "evaluation_metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    workers = 0 if smoke else CONFIG["num_workers"]
    fusion_frame = manifest.loc[manifest["split"] == "fusion"].reset_index(drop=True)
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    fusion = predict(model, fusion_frame, device, workers)
    temperature = fit_temperature(fusion["logits"].to(device), fusion["labels"].to(device))
    test = predict(model, test_frame, device, workers)
    probabilities = torch.softmax(test["logits"].float() / temperature, dim=1)
    metrics = evaluate_probabilities(probabilities, test["labels"])
    metrics["temperature_fitted_on_fusion_split"] = temperature
    metrics["selected_epoch"] = int(torch.load(output_dir / "efficientnet_b0_best.pt", map_location="cpu")["epoch"])
    metrics["trainable_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    metrics["total_parameters"] = int(sum(p.numel() for p in model.parameters()))

    prediction = probabilities.argmax(1).numpy()
    frame = pd.DataFrame({
        "image_id": test["image_ids"],
        "lesion_id": test["lesion_ids"],
        "true_label_index": test["labels"].numpy(),
        "true_label": [CLASS_NAMES[i] for i in test["labels"].numpy()],
        "prediction_index": prediction,
        "prediction": [CLASS_NAMES[i] for i in prediction],
        "correct": prediction == test["labels"].numpy(),
    })
    for index, name in enumerate(CLASS_NAMES):
        frame[f"p_{name}"] = probabilities[:, index].numpy()
    frame.to_csv(output_dir / "test_predictions.csv", index=False)
    atomic_torch_save(
        {
            "logits": test["logits"],
            "probabilities": probabilities,
            "labels": test["labels"],
            "image_ids": test["image_ids"],
            "lesion_ids": test["lesion_ids"],
            "temperature": temperature,
        },
        output_dir / "test_outputs.pt",
    )

    if not smoke:
        original_path = ROOT / "experiment_parts" / "part_04_completed_results_export" / "analysis_and_results" / "held_out_test_predictions.csv"
        original = pd.read_csv(original_path)
        aligned = frame.merge(
            original[["image_id", "lesion_id", "true_label_index"] + [f"baseline_p_{name}" for name in CLASS_NAMES]],
            on=["image_id", "lesion_id", "true_label_index"],
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != len(frame):
            raise AssertionError("Comparator and original test predictions do not align one-to-one")
        candidate = torch.tensor(aligned[[f"p_{name}" for name in CLASS_NAMES]].to_numpy(), dtype=torch.float32)
        baseline = torch.tensor(aligned[[f"baseline_p_{name}" for name in CLASS_NAMES]].to_numpy(), dtype=torch.float32)
        labels = torch.tensor(aligned["true_label_index"].to_numpy(), dtype=torch.long)
        comparison = paired_group_bootstrap(
            candidate,
            baseline,
            labels,
            aligned["lesion_id"].astype(str).tolist(),
            CONFIG["bootstrap_resamples"],
            CONFIG["seed"],
        )
        comparison["estimand"] = "EfficientNet-B0 minus original HardPair-Net generalist pair-balanced accuracy"
        metrics["paired_vs_original_baseline"] = comparison

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(CONFIG["seed"])
    output_dir = PART / "analysis_and_results" / ("comparator_smoke" if args.smoke else "comparator_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "experiment_parts" / "part_01_ham10000_core_experiment" / "analysis_and_results" / "outputs" / "ham10000_splits.csv"
    manifest = pd.read_csv(manifest_path)
    if args.smoke:
        manifest = stratified_smoke_sample(manifest, args.max_images, CONFIG["seed"])
        manifest.to_csv(output_dir / "smoke_manifest.csv", index=False)
    train_frame = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest.loc[manifest["split"] == "val"].reset_index(drop=True)
    model = EfficientNetComparator(pretrained=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model, history = train(model, train_frame, val_frame, output_dir, device, args.smoke)
    training_seconds = time.monotonic() - started
    metrics = evaluate_once(model, manifest, output_dir, device, args.smoke)
    runtime = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_and_evaluation_seconds_current_session": training_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "epochs_in_history": len(history),
        "smoke": args.smoke,
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime": runtime, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
