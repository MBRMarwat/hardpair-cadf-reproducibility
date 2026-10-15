from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import HourlyCheckpoint, atomic_torch_save, capture_rng_state, restore_rng_state
from .data import (
    MEL_IDX,
    NV_IDX,
    ResumableRandomSampler,
    effective_number_class_weights,
    make_loader,
)
from .fusion import SoftFeasibilityFusion, build_gate_inputs
from .losses import (
    ErrorFocusedSpecialistLoss,
    build_feasibility_targets,
    correction_rates,
    specialist_cross_entropy,
)
from .metrics import evaluate_probabilities, pair_balanced_accuracy
from .model import CADFNet, FeasibilityPredictor, MLPGate


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def initialize_components(
    config: dict[str, Any],
) -> tuple[CADFNet, FeasibilityPredictor, MLPGate, SoftFeasibilityFusion]:
    model_config = config["model"]
    model = CADFNet(
        pretrained=bool(model_config["pretrained"]),
        generalist_hidden=int(model_config["generalist_hidden"]),
        specialist_hidden=int(model_config["specialist_hidden"]),
        dropout=float(model_config["dropout"]),
    )
    feasibility = FeasibilityPredictor()
    gate = MLPGate(
        hidden_dim=int(model_config["gate_hidden"]),
        dropout=float(model_config["dropout"]),
        initial_alpha_cap=float(config["fusion"]["alpha_initial_cap"]),
    )
    fusion = SoftFeasibilityFusion(
        kappa=float(config["fusion"]["kappa"]),
        temperature=float(config["fusion"]["temperature"]),
    )
    return model, feasibility, gate, fusion


def _normal_cpu(tensor: torch.Tensor) -> torch.Tensor:
    # torch.no_grad (rather than inference_mode) keeps cache tensors autograd-compatible.
    return tensor.detach().to("cpu").clone()


def cache_initial_features(
    model: CADFNet,
    frame: pd.DataFrame,
    destination: str | Path,
    config: dict[str, Any],
    device: torch.device,
    same_view: bool = False,
) -> dict[str, Any]:
    destination = Path(destination)
    if destination.is_file():
        cached = torch.load(destination, map_location="cpu")
        if cached.get("complete") and len(cached["labels"]) == len(frame):
            return cached
    progress_path = destination.with_suffix(".progress.pt")
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu")
    else:
        progress = {
            "next_index": 0,
            "generalist_features": [],
            "specialist_features": [],
            "color_stats": [],
            "labels": [],
            "image_ids": [],
            "lesion_ids": [],
            "splits": [],
        }
    start = int(progress["next_index"])
    loader = make_loader(
        frame.iloc[start:].reset_index(drop=True),
        config,
        training=False,
        batch_size=int(config["data"]["feature_batch_size"]),
        same_view=same_view,
    )
    checkpoint = HourlyCheckpoint(
        progress_path, float(config["training"]["checkpoint_interval_seconds"])
    )
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            generalist = batch["generalist_view"].to(device, non_blocking=True)
            specialist = batch["specialist_view"].to(device, non_blocking=True)
            progress["generalist_features"].append(_normal_cpu(model.encode(generalist)))
            progress["specialist_features"].append(_normal_cpu(model.encode(specialist)))
            progress["color_stats"].append(_normal_cpu(batch["color_stats"]))
            progress["labels"].append(_normal_cpu(batch["label"]))
            progress["image_ids"].extend(list(batch["image_id"]))
            progress["lesion_ids"].extend(list(batch["lesion_id"]))
            count = int(len(batch["label"]))
            progress["splits"].extend(frame.iloc[start : start + count]["split"].astype(str).tolist())
            start += count
            progress["next_index"] = start
            if checkpoint.due():
                checkpoint.save(progress)
    complete = {
        "complete": True,
        "generalist_features": torch.cat(progress["generalist_features"], dim=0),
        "specialist_features": torch.cat(progress["specialist_features"], dim=0),
        "color_stats": torch.cat(progress["color_stats"], dim=0),
        "labels": torch.cat(progress["labels"], dim=0).long(),
        "image_ids": progress["image_ids"],
        "lesion_ids": progress["lesion_ids"],
        "splits": progress["splits"],
        "same_view": bool(same_view),
    }
    atomic_torch_save(complete, destination)
    if progress_path.exists():
        progress_path.unlink()
    return complete


def _checkpoint_payload(
    stage: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    next_sample: int,
    history: list[dict[str, float]],
    complete: bool,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "next_sample": int(next_sample),
        "history": history,
        "complete": bool(complete),
        "best_metric": float(best_metric),
        "rng_state": capture_rng_state(),
    }


def _head_validation(model: CADFNet, cache: dict[str, Any], split: str) -> dict[str, float]:
    mask = torch.tensor([value == split for value in cache["splits"]], dtype=torch.bool)
    with torch.no_grad():
        generalist_logits = model.generalist_from_features(cache["generalist_features"][mask])
        specialist_logits = model.specialist_from_features(cache["specialist_features"][mask])
        probabilities = F.softmax(generalist_logits, dim=1)
    prediction = probabilities.argmax(dim=1).numpy()
    labels = cache["labels"][mask].numpy()
    macro_f1 = float(f1_score(labels, prediction, labels=range(7), average="macro", zero_division=0))
    rates = correction_rates(specialist_logits, generalist_logits, cache["labels"][mask])
    return {"macro_f1": macro_f1, **rates}


def train_stage1_generalist(
    model: CADFNet,
    cache: dict[str, Any],
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    latest = output_dir / "stage1_generalist_latest.pt"
    best = output_dir / "stage1_generalist_best.pt"
    model.freeze_encoder()
    for parameter in model.specialist_head.parameters():
        parameter.requires_grad = False
    for parameter in model.generalist_head.parameters():
        parameter.requires_grad = True
    model.to(device)

    train_mask = torch.tensor([value == "train" for value in cache["splits"]], dtype=torch.bool)
    features = cache["generalist_features"][train_mask]
    labels = cache["labels"][train_mask]
    class_weights = effective_number_class_weights(
        labels, 7, float(config["training"]["effective_number_beta"])
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.generalist_head.parameters(),
        lr=float(config["training"]["stage1_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["stage1_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, float]] = []
    start_epoch = start_sample = 0
    best_metric = float("-inf")
    if latest.is_file():
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model_state"])
        if state.get("complete"):
            if best.is_file():
                model.load_state_dict(torch.load(best, map_location=device)["model_state"])
            return state["history"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch, start_sample = int(state["epoch"]), int(state["next_sample"])
        best_metric = float(state["best_metric"])
        restore_rng_state(state.get("rng_state"))
    checkpoint = HourlyCheckpoint(latest, config["training"]["checkpoint_interval_seconds"])
    batch_size = int(config["data"]["feature_batch_size"])
    dataset = TensorDataset(features, labels)
    for epoch in range(start_epoch, epochs):
        epoch_start = start_sample if epoch == start_epoch else 0
        sampler = ResumableRandomSampler(dataset, config["seed"], epoch, epoch_start)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        model.generalist_head.train()
        running_loss = examples = 0
        next_sample = epoch_start
        for feature_batch, label_batch in loader:
            feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
            loss = criterion(model.generalist_from_features(feature_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.generalist_head.parameters(), config["training"]["gradient_clip_norm"]
            )
            optimizer.step()
            count = len(label_batch)
            running_loss += float(loss) * count
            examples += count
            next_sample += count
            if checkpoint.due():
                checkpoint.save(
                    _checkpoint_payload(
                        "stage1", model, optimizer, scheduler, epoch, next_sample,
                        history, False, best_metric,
                    )
                )
        scheduler.step()
        validation = _head_validation(model.to("cpu"), cache, "val")
        model.to(device)
        record = {"epoch": float(epoch + 1), "loss": running_loss / max(1, examples), **validation}
        history.append(record)
        if validation["macro_f1"] > best_metric:
            best_metric = validation["macro_f1"]
            atomic_torch_save({"model_state": model.state_dict(), "record": record}, best)
        checkpoint.save(
            _checkpoint_payload(
                "stage1", model, optimizer, scheduler, epoch + 1, 0, history,
                epoch + 1 >= epochs, best_metric,
            )
        )
        start_sample = 0
    model.load_state_dict(torch.load(best, map_location=device)["model_state"])
    return history


def train_stage2_specialist(
    model: CADFNet,
    cache: dict[str, Any],
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
    standard_cross_entropy: bool = False,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    suffix = "ce" if standard_cross_entropy else "error_focused"
    latest = output_dir / f"stage2_specialist_{suffix}_latest.pt"
    best = output_dir / f"stage2_specialist_{suffix}_best.pt"
    model.freeze_encoder()
    for parameter in model.generalist_head.parameters():
        parameter.requires_grad = False
    for parameter in model.specialist_head.parameters():
        parameter.requires_grad = True
    model.to(device)
    train_mask = torch.tensor(
        [split == "train" for split in cache["splits"]], dtype=torch.bool
    ) & ((cache["labels"] == MEL_IDX) | (cache["labels"] == NV_IDX))
    specialist_features = cache["specialist_features"][train_mask]
    generalist_features = cache["generalist_features"][train_mask]
    labels = cache["labels"][train_mask]
    targets = (labels == NV_IDX).long()
    dataset = TensorDataset(specialist_features, generalist_features, targets)
    if standard_cross_entropy:
        criterion: Callable[..., torch.Tensor] = specialist_cross_entropy
    else:
        criterion = ErrorFocusedSpecialistLoss(**config["specialist_loss"])
    optimizer = torch.optim.AdamW(
        model.specialist_head.parameters(),
        lr=float(config["training"]["stage2_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["stage2_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, float]] = []
    start_epoch = start_sample = 0
    best_metric = float("-inf")
    if latest.is_file():
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model_state"])
        if state.get("complete"):
            if best.is_file():
                model.load_state_dict(torch.load(best, map_location=device)["model_state"])
            return state["history"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch, start_sample = int(state["epoch"]), int(state["next_sample"])
        best_metric = float(state["best_metric"])
        restore_rng_state(state.get("rng_state"))
    checkpoint = HourlyCheckpoint(latest, config["training"]["checkpoint_interval_seconds"])
    batch_size = int(config["data"]["feature_batch_size"])
    for epoch in range(start_epoch, epochs):
        epoch_start = start_sample if epoch == start_epoch else 0
        sampler = ResumableRandomSampler(dataset, config["seed"] + 1000, epoch, epoch_start)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        model.specialist_head.train()
        running_loss = examples = 0
        next_sample = epoch_start
        for specialist_batch, generalist_batch, target_batch in loader:
            specialist_batch = specialist_batch.to(device)
            generalist_batch = generalist_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                generalist_probs = F.softmax(
                    model.generalist_from_features(generalist_batch), dim=1
                )
            logits = model.specialist_from_features(specialist_batch)
            loss = criterion(logits, target_batch, generalist_probs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.specialist_head.parameters(), config["training"]["gradient_clip_norm"]
            )
            optimizer.step()
            count = len(target_batch)
            running_loss += float(loss) * count
            examples += count
            next_sample += count
            if checkpoint.due():
                checkpoint.save(
                    _checkpoint_payload(
                        "stage2", model, optimizer, scheduler, epoch, next_sample,
                        history, False, best_metric,
                    )
                )
        scheduler.step()
        validation = _head_validation(model.to("cpu"), cache, "val")
        model.to(device)
        metric = validation["full_error_correction_potential"]
        record = {"epoch": float(epoch + 1), "loss": running_loss / max(1, examples), **validation}
        history.append(record)
        if metric > best_metric:
            best_metric = metric
            atomic_torch_save({"model_state": model.state_dict(), "record": record}, best)
        checkpoint.save(
            _checkpoint_payload(
                "stage2", model, optimizer, scheduler, epoch + 1, 0, history,
                epoch + 1 >= epochs, best_metric,
            )
        )
        start_sample = 0
    model.load_state_dict(torch.load(best, map_location=device)["model_state"])
    return history


def _evaluate_image_model(
    model: CADFNet,
    frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    same_view: bool = False,
) -> dict[str, float]:
    loader = make_loader(
        frame,
        config,
        training=False,
        batch_size=int(config["data"]["evaluation_batch_size"]),
        same_view=same_view,
    )
    model.eval()
    logits_list: list[torch.Tensor] = []
    specialist_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            generalist_logits, specialist_logits = model.forward_views(
                batch["generalist_view"].to(device, non_blocking=True),
                batch["specialist_view"].to(device, non_blocking=True),
            )
            logits_list.append(_normal_cpu(generalist_logits))
            specialist_list.append(_normal_cpu(specialist_logits))
            labels_list.append(_normal_cpu(batch["label"]).long())
    generalist_logits = torch.cat(logits_list)
    specialist_logits = torch.cat(specialist_list)
    labels = torch.cat(labels_list)
    probability_metrics = evaluate_probabilities(F.softmax(generalist_logits, dim=1), labels)
    return {
        "macro_f1": float(probability_metrics["macro_f1"]),
        "mel_sensitivity": float(probability_metrics["mel_sensitivity"]),
        **correction_rates(specialist_logits, generalist_logits, labels),
    }


def train_stage3_finetune(
    model: CADFNet,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
    standard_cross_entropy: bool = False,
    same_view: bool = False,
    checkpoint_prefix: str = "stage3_finetune",
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    latest = output_dir / f"{checkpoint_prefix}_latest.pt"
    best = output_dir / f"{checkpoint_prefix}_best.pt"
    model.unfreeze_last_blocks(int(config["model"]["unfreeze_last_blocks"]))
    for parameter in model.generalist_head.parameters():
        parameter.requires_grad = True
    for parameter in model.specialist_head.parameters():
        parameter.requires_grad = True
    model.to(device)
    model.freeze_batchnorm_statistics()
    train_labels = torch.as_tensor(train_frame["label"].to_numpy(), dtype=torch.long)
    class_weights = effective_number_class_weights(
        train_labels, 7, float(config["training"]["effective_number_beta"])
    ).to(device)
    generalist_criterion = nn.CrossEntropyLoss(weight=class_weights)
    specialist_criterion: Callable[..., torch.Tensor]
    specialist_criterion = (
        specialist_cross_entropy
        if standard_cross_entropy
        else ErrorFocusedSpecialistLoss(**config["specialist_loss"])
    )
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(("generalist_head", "specialist_head"))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(config["training"]["stage3_encoder_lr"])},
            {
                "params": list(model.generalist_head.parameters())
                + list(model.specialist_head.parameters()),
                "lr": float(config["training"]["stage3_head_lr"]),
            },
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["stage3_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, float]] = []
    start_epoch = start_sample = 0
    best_metric = float("-inf")
    if latest.is_file():
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model_state"])
        if state.get("complete"):
            if best.is_file():
                model.load_state_dict(torch.load(best, map_location=device)["model_state"])
            return state["history"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch, start_sample = int(state["epoch"]), int(state["next_sample"])
        best_metric = float(state["best_metric"])
        restore_rng_state(state.get("rng_state"))
    checkpoint = HourlyCheckpoint(latest, config["training"]["checkpoint_interval_seconds"])
    batch_size = int(config["data"]["image_batch_size"])
    for epoch in range(start_epoch, epochs):
        epoch_start = start_sample if epoch == start_epoch else 0
        loader = make_loader(
            train_frame,
            config,
            training=True,
            batch_size=batch_size,
            epoch=epoch,
            start_index=epoch_start,
            same_view=same_view,
        )
        model.train()
        model.freeze_batchnorm_statistics()
        running_loss = generalist_total = specialist_total = 0.0
        examples = specialist_examples = 0
        next_sample = epoch_start
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            generalist_logits, specialist_logits = model.forward_views(
                batch["generalist_view"].to(device, non_blocking=True),
                batch["specialist_view"].to(device, non_blocking=True),
            )
            generalist_loss = generalist_criterion(generalist_logits, labels)
            pair_mask = (labels == MEL_IDX) | (labels == NV_IDX)
            if pair_mask.any():
                specialist_targets = (labels[pair_mask] == NV_IDX).long()
                generalist_probs = F.softmax(generalist_logits[pair_mask].detach(), dim=1)
                specialist_loss = specialist_criterion(
                    specialist_logits[pair_mask], specialist_targets, generalist_probs
                )
                pair_count = int(pair_mask.sum())
            else:
                specialist_loss = specialist_logits.sum() * 0.0
                pair_count = 0
            loss = generalist_loss + float(config["training"]["specialist_loss_weight"]) * specialist_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config["training"]["gradient_clip_norm"],
            )
            optimizer.step()
            count = len(labels)
            running_loss += float(loss) * count
            generalist_total += float(generalist_loss) * count
            specialist_total += float(specialist_loss) * pair_count
            examples += count
            specialist_examples += pair_count
            next_sample += count
            if checkpoint.due():
                checkpoint.save(
                    _checkpoint_payload(
                        "stage3", model, optimizer, scheduler, epoch, next_sample,
                        history, False, best_metric,
                    )
                )
        scheduler.step()
        validation = _evaluate_image_model(
            model, validation_frame, config, device, same_view=same_view
        )
        record = {
            "epoch": float(epoch + 1),
            "loss": running_loss / max(1, examples),
            "generalist_loss": generalist_total / max(1, examples),
            "specialist_loss": specialist_total / max(1, specialist_examples),
            **validation,
        }
        history.append(record)
        selection_metric = validation["macro_f1"] + 0.25 * validation["mel_sensitivity"]
        if selection_metric > best_metric:
            best_metric = selection_metric
            atomic_torch_save({"model_state": model.state_dict(), "record": record}, best)
        checkpoint.save(
            _checkpoint_payload(
                "stage3", model, optimizer, scheduler, epoch + 1, 0, history,
                epoch + 1 >= epochs, best_metric,
            )
        )
        start_sample = 0
    model.load_state_dict(torch.load(best, map_location=device)["model_state"])
    return history


def collect_outputs(
    model: CADFNet,
    frame: pd.DataFrame,
    destination: str | Path,
    config: dict[str, Any],
    device: torch.device,
    same_view: bool = False,
) -> dict[str, Any]:
    destination = Path(destination)
    if destination.is_file():
        saved = torch.load(destination, map_location="cpu")
        if saved.get("complete") and len(saved["labels"]) == len(frame):
            return saved
    progress_path = destination.with_suffix(".progress.pt")
    progress = (
        torch.load(progress_path, map_location="cpu")
        if progress_path.is_file()
        else {
            "next_index": 0,
            "generalist_logits": [],
            "specialist_logits": [],
            "color_stats": [],
            "labels": [],
            "image_ids": [],
            "lesion_ids": [],
        }
    )
    start = int(progress["next_index"])
    loader = make_loader(
        frame.iloc[start:].reset_index(drop=True),
        config,
        training=False,
        batch_size=int(config["data"]["evaluation_batch_size"]),
        same_view=same_view,
    )
    checkpoint = HourlyCheckpoint(
        progress_path, config["training"]["checkpoint_interval_seconds"]
    )
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            generalist_logits, specialist_logits = model.forward_views(
                batch["generalist_view"].to(device, non_blocking=True),
                batch["specialist_view"].to(device, non_blocking=True),
            )
            progress["generalist_logits"].append(_normal_cpu(generalist_logits))
            progress["specialist_logits"].append(_normal_cpu(specialist_logits))
            progress["color_stats"].append(_normal_cpu(batch["color_stats"]))
            progress["labels"].append(_normal_cpu(batch["label"]).long())
            progress["image_ids"].extend(list(batch["image_id"]))
            progress["lesion_ids"].extend(list(batch["lesion_id"]))
            count = len(batch["label"])
            start += count
            progress["next_index"] = start
            if checkpoint.due():
                checkpoint.save(progress)
    complete = {
        "complete": True,
        "generalist_logits": torch.cat(progress["generalist_logits"]),
        "specialist_logits": torch.cat(progress["specialist_logits"]),
        "color_stats": torch.cat(progress["color_stats"]),
        "labels": torch.cat(progress["labels"]).long(),
        "image_ids": progress["image_ids"],
        "lesion_ids": progress["lesion_ids"],
        "same_view": bool(same_view),
    }
    atomic_torch_save(complete, destination)
    if progress_path.exists():
        progress_path.unlink()
    return complete


def train_stage4_feasibility(
    predictor: FeasibilityPredictor,
    fusion_outputs: dict[str, Any],
    validation_outputs: dict[str, Any],
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    latest = output_dir / "stage4_feasibility_latest.pt"
    best = output_dir / "stage4_feasibility_best.pt"
    predictor.to(device)
    generalist_predictions = fusion_outputs["generalist_logits"].argmax(dim=1)
    targets = build_feasibility_targets(fusion_outputs["labels"], generalist_predictions)
    positive = targets.sum(dim=0)
    negative = len(targets) - positive
    positive_weight = torch.where(positive > 0, negative / positive.clamp_min(1.0), torch.ones_like(positive))
    positive_weight = positive_weight.clamp(1.0, 50.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    dataset = TensorDataset(fusion_outputs["color_stats"], targets)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["training"]["stage4_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["stage4_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, float]] = []
    start_epoch = start_sample = 0
    best_metric = float("-inf")
    if latest.is_file():
        state = torch.load(latest, map_location=device)
        predictor.load_state_dict(state["model_state"])
        if state.get("complete"):
            if best.is_file():
                predictor.load_state_dict(torch.load(best, map_location=device)["model_state"])
            return state["history"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch, start_sample = int(state["epoch"]), int(state["next_sample"])
        best_metric = float(state["best_metric"])
        restore_rng_state(state.get("rng_state"))
    checkpoint = HourlyCheckpoint(latest, config["training"]["checkpoint_interval_seconds"])
    batch_size = max(16, int(config["data"]["feature_batch_size"]))
    validation_targets = build_feasibility_targets(
        validation_outputs["labels"], validation_outputs["generalist_logits"].argmax(dim=1)
    )
    for epoch in range(start_epoch, epochs):
        epoch_start = start_sample if epoch == start_epoch else 0
        sampler = ResumableRandomSampler(dataset, config["seed"] + 2000, epoch, epoch_start)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        predictor.train()
        running_loss = examples = 0
        next_sample = epoch_start
        for stats, target in loader:
            stats, target = stats.to(device), target.to(device)
            loss = criterion(predictor(stats, return_logits=True), target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), config["training"]["gradient_clip_norm"])
            optimizer.step()
            count = len(target)
            running_loss += float(loss) * count
            examples += count
            next_sample += count
            if checkpoint.due():
                checkpoint.save(
                    _checkpoint_payload(
                        "stage4", predictor, optimizer, scheduler, epoch, next_sample,
                        history, False, best_metric,
                    )
                )
        scheduler.step()
        predictor.eval()
        with torch.no_grad():
            validation_logits = predictor(
                validation_outputs["color_stats"].to(device), return_logits=True
            )
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    validation_logits, validation_targets.to(device), pos_weight=positive_weight
                )
            )
            validation_probability = torch.sigmoid(validation_logits)
            positive_mask = validation_targets.to(device) > 0.5
            negative_mask = ~positive_mask
            positive_mean = float(validation_probability[positive_mask].mean()) if positive_mask.any() else 0.0
            negative_mean = float(validation_probability[negative_mask].mean()) if negative_mask.any() else 0.0
            separation = positive_mean - negative_mean
        record = {
            "epoch": float(epoch + 1),
            "loss": running_loss / max(1, examples),
            "validation_loss": validation_loss,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "separation": separation,
        }
        history.append(record)
        metric = -validation_loss
        if metric > best_metric:
            best_metric = metric
            atomic_torch_save({"model_state": predictor.state_dict(), "record": record}, best)
        checkpoint.save(
            _checkpoint_payload(
                "stage4", predictor, optimizer, scheduler, epoch + 1, 0, history,
                epoch + 1 >= epochs, best_metric,
            )
        )
        start_sample = 0
    predictor.load_state_dict(torch.load(best, map_location=device)["model_state"])
    return history


def probabilities_from_outputs(outputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        F.softmax(outputs["generalist_logits"].float(), dim=1),
        F.softmax(outputs["specialist_logits"].float(), dim=1),
    )


def fuse_outputs(
    outputs: dict[str, Any],
    predictor: FeasibilityPredictor,
    gate: MLPGate,
    fusion: SoftFeasibilityFusion,
    device: torch.device,
    alpha_override: float | None = None,
    beta_override: float | None = None,
) -> dict[str, torch.Tensor]:
    probabilities, specialist = probabilities_from_outputs(outputs)
    probabilities = probabilities.to(device)
    specialist = specialist.to(device)
    stats = outputs["color_stats"].to(device)
    predictor.eval()
    gate.eval()
    with torch.no_grad():
        feasibility = predictor(stats)
        gate_inputs = build_gate_inputs(probabilities, specialist, feasibility)
        alpha, beta = gate(gate_inputs)
        if alpha_override is not None:
            alpha = torch.full_like(alpha, float(alpha_override))
        if beta_override is not None:
            beta = torch.full_like(beta, float(beta_override))
        fused = fusion(probabilities, specialist, alpha, beta, feasibility)
    return {
        "probabilities": _normal_cpu(probabilities),
        "specialist": _normal_cpu(specialist),
        "feasibility": _normal_cpu(feasibility),
        "alpha": _normal_cpu(alpha),
        "beta": _normal_cpu(beta),
        "fused": _normal_cpu(fused),
    }


def train_stage5_gate(
    predictor: FeasibilityPredictor,
    gate: MLPGate,
    fusion: SoftFeasibilityFusion,
    fusion_outputs: dict[str, Any],
    validation_outputs: dict[str, Any],
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    latest = output_dir / "stage5_gate_latest.pt"
    best = output_dir / "stage5_gate_best.pt"
    predictor.to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad = False
    gate.to(device)
    probabilities, specialist = probabilities_from_outputs(fusion_outputs)
    with torch.no_grad():
        feasibility = _normal_cpu(predictor(fusion_outputs["color_stats"].to(device)))
    dataset = TensorDataset(
        probabilities, specialist, feasibility, fusion_outputs["labels"]
    )
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(config["training"]["stage5_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["stage5_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, float]] = []
    start_epoch = start_sample = 0
    best_metric = float("-inf")
    if latest.is_file():
        state = torch.load(latest, map_location=device)
        gate.load_state_dict(state["model_state"])
        if state.get("complete"):
            if best.is_file():
                gate.load_state_dict(torch.load(best, map_location=device)["model_state"])
            return state["history"]
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch, start_sample = int(state["epoch"]), int(state["next_sample"])
        best_metric = float(state["best_metric"])
        restore_rng_state(state.get("rng_state"))
    checkpoint = HourlyCheckpoint(latest, config["training"]["checkpoint_interval_seconds"])
    batch_size = int(config["data"]["feature_batch_size"])
    for epoch in range(start_epoch, epochs):
        epoch_start = start_sample if epoch == start_epoch else 0
        sampler = ResumableRandomSampler(dataset, config["seed"] + 3000, epoch, epoch_start)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        gate.train()
        running_loss = examples = 0
        next_sample = epoch_start
        for probability_batch, specialist_batch, feasibility_batch, label_batch in loader:
            probability_batch = probability_batch.to(device)
            specialist_batch = specialist_batch.to(device)
            feasibility_batch = feasibility_batch.to(device)
            label_batch = label_batch.to(device)
            gate_inputs = build_gate_inputs(
                probability_batch, specialist_batch, feasibility_batch
            )
            alpha, beta = gate(gate_inputs)
            fused = fusion(
                probability_batch, specialist_batch, alpha, beta, feasibility_batch
            )
            nll = F.nll_loss(fused.clamp_min(1e-8).log(), label_batch)
            offpair_mask = (label_batch != MEL_IDX) & (label_batch != NV_IDX)
            safety = (
                (fused[offpair_mask, MEL_IDX] - probability_batch[offpair_mask, MEL_IDX])
                .clamp_min(0.0)
                .mean()
                if offpair_mask.any()
                else fused.sum() * 0.0
            )
            loss = nll + float(config["training"]["offpair_safety_weight"]) * safety
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), config["training"]["gradient_clip_norm"])
            optimizer.step()
            count = len(label_batch)
            running_loss += float(loss) * count
            examples += count
            next_sample += count
            if checkpoint.due():
                checkpoint.save(
                    _checkpoint_payload(
                        "stage5", gate, optimizer, scheduler, epoch, next_sample,
                        history, False, best_metric,
                    )
                )
        scheduler.step()
        validation = fuse_outputs(
            validation_outputs, predictor, gate, fusion, device
        )
        validation_metric = pair_balanced_accuracy(
            validation["fused"], validation_outputs["labels"]
        )
        record = {
            "epoch": float(epoch + 1),
            "loss": running_loss / max(1, examples),
            "validation_pair_balanced_accuracy": validation_metric,
            "mean_alpha": float(validation["alpha"].mean()),
            "mean_beta": float(validation["beta"].mean()),
        }
        history.append(record)
        if validation_metric > best_metric:
            best_metric = validation_metric
            atomic_torch_save({"model_state": gate.state_dict(), "record": record}, best)
        checkpoint.save(
            _checkpoint_payload(
                "stage5", gate, optimizer, scheduler, epoch + 1, 0, history,
                epoch + 1 >= epochs, best_metric,
            )
        )
        start_sample = 0
    gate.load_state_dict(torch.load(best, map_location=device)["model_state"])
    return history


def evaluate_ablation_variants(
    outputs: dict[str, Any],
    predictor: FeasibilityPredictor,
    gate: MLPGate,
    fusion: SoftFeasibilityFusion,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    baseline, _ = probabilities_from_outputs(outputs)
    variants: dict[str, dict[str, torch.Tensor]] = {
        "A0_full": fuse_outputs(outputs, predictor, gate, fusion, device),
        "A3_no_feasibility_transfer": fuse_outputs(
            outputs, predictor, gate, fusion, device, beta_override=0.0
        ),
        "A4_fixed_gate": fuse_outputs(
            outputs,
            predictor,
            gate,
            fusion,
            device,
            alpha_override=float(config["fusion"]["fixed_alpha"]),
            beta_override=float(config["fusion"]["fixed_beta"]),
        ),
    }
    metrics = {
        "A1_generalist_only": evaluate_probabilities(baseline, outputs["labels"])
    }
    for name, values in variants.items():
        metrics[name] = evaluate_probabilities(
            values["fused"], outputs["labels"], reference=baseline
        )
    return metrics, variants


def write_histories(
    output_dir: str | Path, histories: dict[str, list[dict[str, float]]]
) -> None:
    destination = Path(output_dir) / "training_histories.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(histories, indent=2, allow_nan=False), encoding="utf-8")
