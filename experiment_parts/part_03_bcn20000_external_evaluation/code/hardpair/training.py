from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .checkpoint import CheckpointManager, atomic_torch_save, capture_rng_state, restore_rng_state
from .config import seed_everything
from .data import (
    CLASS_NAMES,
    MEL_IDX,
    NV_IDX,
    HAM10000Dataset,
    ResumableRandomSampler,
    build_transform,
    seed_worker,
)
from .model import HardPairNet


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=len(CLASS_NAMES)).float().clamp_min(1)
    weights = (counts[NV_IDX] / counts).pow(0.35).clamp_max(3.0)
    return weights


def _loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    workers = int(config["dataset"]["num_workers"])
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }


def extract_feature_cache(
    model: HardPairNet,
    manifest: pd.DataFrame,
    cache_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Extract deterministic features with hourly, resumable progress snapshots."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("complete") and cached.get("image_ids") == manifest.image_id.tolist():
            return cached
    interval = float(config["training"]["checkpoint_interval_seconds"])
    progress_path = cache_path.with_suffix(".progress.pt")
    manager = CheckpointManager(progress_path, interval)
    features = torch.empty((len(manifest), model.feature_dim), dtype=torch.float32)
    completed = 0
    if manager.exists:
        progress = manager.load("cpu")
        if progress.get("image_ids") == manifest.image_id.tolist():
            completed = int(progress["completed"])
            features[:completed] = progress["features"][:completed]

    dataset = HAM10000Dataset(
        manifest, build_transform(int(config["dataset"]["image_size"]), training=False)
    )
    remaining = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        remaining,
        batch_size=max(1, int(config["training"]["batch_size"]) * 2),
        shuffle=False,
        **_loader_kwargs(config, device),
    )
    model = model.to(device).eval()
    cursor = completed
    with torch.inference_mode():
        for batch in loader:
            values = model.extract_features(batch["image"].to(device, non_blocking=True)).cpu()
            features[cursor : cursor + len(values)] = values
            cursor += len(values)
            if manager.due():
                manager.save(
                    {
                        "complete": False,
                        "completed": cursor,
                        "features": features,
                        "image_ids": manifest.image_id.tolist(),
                    }
                )
    payload = {
        "complete": True,
        "features": features,
        "labels": torch.tensor(manifest.label.to_numpy(), dtype=torch.long),
        "image_ids": manifest.image_id.tolist(),
        "lesion_ids": manifest.lesion_id.tolist(),
        "splits": manifest.split.tolist(),
    }
    atomic_torch_save(payload, cache_path)
    if progress_path.exists():
        progress_path.unlink()
    return payload


def _cached_split(cache: dict[str, Any], split: str) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor([value == split for value in cache["splits"]])
    # clone() outside inference_mode strips any inference-only marker from
    # caches created by older versions of the extractor.
    return cache["features"][mask].detach().clone(), cache["labels"][mask].detach().clone()


@torch.inference_mode()
def _score_feature_generalist(
    model: HardPairNet, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[float, float]:
    model.eval()
    logits, _ = model.heads_from_features(features.to(device))
    predictions = logits.argmax(1).cpu().numpy()
    targets = labels.numpy()
    macro_f1 = float(f1_score(targets, predictions, labels=range(7), average="macro", zero_division=0))
    mel_mask = targets == MEL_IDX
    mel_sensitivity = float((predictions[mel_mask] == MEL_IDX).mean()) if mel_mask.any() else 0.0
    return macro_f1, mel_sensitivity


def initialize_heads(
    model: HardPairNet,
    cache: dict[str, Any],
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state.get("complete"):
            model.load_state_dict(state["model_state"])
            return state
    train_x, train_y = _cached_split(cache, "train")
    val_x, val_y = _cached_split(cache, "val")
    model = model.to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    cfg = config["training"]
    batch_size = int(cfg["head_batch_size"])
    max_epochs = int(cfg["max_head_epochs"])
    patience = int(cfg["head_patience"])
    smoke = bool(config.get("_smoke", False))
    if smoke:
        max_epochs, patience = min(max_epochs, 2), 1

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y).to(device))
    optimizer = torch.optim.AdamW(
        list(model.norm.parameters()) + list(model.generalist.parameters()),
        lr=float(cfg["frozen_head_lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_key = (-float("inf"), -float("inf"))
    best_state = None
    stale = 0
    for epoch in range(max_epochs):
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, generator=generator)
        model.norm.train()
        model.generalist.train()
        for features, labels in loader:
            logits, _ = model.heads_from_features(features.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        key = _score_feature_generalist(model, val_x, val_y, device)
        if key > best_key:
            best_key = key
            best_state = {
                "norm": copy.deepcopy(model.norm.state_dict()),
                "generalist": copy.deepcopy(model.generalist.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        atomic_torch_save(
            {"stage": "generalist", "epoch": epoch + 1, "best_key": best_key, "model_state": model.state_dict()},
            checkpoint_path.with_suffix(".progress.pt"),
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Generalist head initialization produced no checkpoint")
    model.norm.load_state_dict(best_state["norm"])
    model.generalist.load_state_dict(best_state["generalist"])

    for parameter in model.norm.parameters():
        parameter.requires_grad = False
    pair_mask = (train_y == MEL_IDX) | (train_y == NV_IDX)
    pair_val_mask = (val_y == MEL_IDX) | (val_y == NV_IDX)
    pair_x = train_x[pair_mask]
    pair_y = (train_y[pair_mask] == NV_IDX).long()
    pair_val_x = val_x[pair_val_mask]
    pair_val_y = (val_y[pair_val_mask] == NV_IDX).long()
    optimizer = torch.optim.AdamW(
        model.specialist.parameters(), lr=float(cfg["frozen_head_lr"]), weight_decay=float(cfg["weight_decay"])
    )
    best_accuracy = -1.0
    best_specialist = None
    stale = 0
    for epoch in range(max_epochs):
        generator = torch.Generator().manual_seed(int(config["seed"]) + 10_000 + epoch)
        loader = DataLoader(TensorDataset(pair_x, pair_y), batch_size=batch_size, shuffle=True, generator=generator)
        model.specialist.train()
        for features, labels in loader:
            normalized = model.norm(features.to(device))
            logits = model.specialist(normalized)
            loss = nn.functional.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            normalized = model.norm(pair_val_x.to(device))
            accuracy = float((model.specialist(normalized).argmax(1).cpu() == pair_val_y).float().mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_specialist = copy.deepcopy(model.specialist.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_specialist is None:
        raise RuntimeError("Specialist head initialization produced no checkpoint")
    model.specialist.load_state_dict(best_specialist)
    state = {
        "complete": True,
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "generalist_validation_macro_f1": best_key[0],
        "generalist_validation_mel_sensitivity": best_key[1],
        "specialist_validation_accuracy": best_accuracy,
    }
    atomic_torch_save(state, checkpoint_path)
    progress_path = checkpoint_path.with_suffix(".progress.pt")
    if progress_path.exists():
        progress_path.unlink()
    return state


@torch.inference_mode()
def predict_model(
    model: HardPairNet,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    dataset = HAM10000Dataset(
        manifest, build_transform(int(config["dataset"]["image_size"]), training=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config["training"]["batch_size"]) * 2),
        shuffle=False,
        **_loader_kwargs(config, device),
    )
    model = model.to(device).eval()
    generalist: list[torch.Tensor] = []
    specialist: list[torch.Tensor] = []
    for batch in loader:
        logits_7, logits_2 = model(batch["image"].to(device, non_blocking=True))
        generalist.append(logits_7.cpu())
        specialist.append(logits_2.cpu())
    return {
        "generalist_logits": torch.cat(generalist),
        "specialist_logits": torch.cat(specialist),
        "labels": torch.tensor(manifest.label.to_numpy(), dtype=torch.long),
        "image_ids": manifest.image_id.tolist(),
        "lesion_ids": manifest.lesion_id.tolist(),
    }


def _validation_score(model: HardPairNet, frame: pd.DataFrame, config: dict, device: torch.device) -> tuple[float, float]:
    output = predict_model(model, frame, config, device)
    truth = output["labels"].numpy()
    predictions = output["generalist_logits"].argmax(1).numpy()
    macro_f1 = float(f1_score(truth, predictions, labels=range(7), average="macro", zero_division=0))
    mel_mask = truth == MEL_IDX
    mel_sensitivity = float((predictions[mel_mask] == MEL_IDX).mean()) if mel_mask.any() else 0.0
    return macro_f1, mel_sensitivity


def train_image_model(
    model: HardPairNet,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    run_name: str,
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
    multitask: bool,
) -> tuple[HardPairNet, dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / f"{run_name}_latest.pt"
    best_path = output_dir / f"{run_name}_best.pt"
    manager = CheckpointManager(latest_path, float(config["training"]["checkpoint_interval_seconds"]))
    # Both ablations receive the same data order, worker seeds, and augmentation
    # stream when starting fresh. A resumed run restores its saved RNG state below.
    if not manager.exists:
        seed_everything(int(config["seed"]))
    cfg = config["training"]
    max_epochs = int(cfg["max_finetune_epochs"])
    min_epochs = int(cfg["min_finetune_epochs"])
    patience = int(cfg["finetune_patience"])
    if config.get("_smoke", False):
        max_epochs, min_epochs, patience = 1, 1, 1

    model.freeze_for_partial_finetuning(int(config["model"]["unfreeze_last_modules"]))
    backbone_parameters = [parameter for parameter in model.features.parameters() if parameter.requires_grad]
    head_parameters = list(model.norm.parameters()) + list(model.generalist.parameters())
    if multitask:
        head_parameters += list(model.specialist.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(cfg["backbone_lr"])},
            {"params": head_parameters, "lr": float(cfg["head_lr"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    train_labels = torch.tensor(train_frame.label.to_numpy(), dtype=torch.long)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels).to(device))
    dataset = HAM10000Dataset(
        train_frame, build_transform(int(config["dataset"]["image_size"]), training=True)
    )

    epoch, seen = 0, 0
    loss_sum, loss_count = 0.0, 0
    best_key = (-float("inf"), -float("inf"))
    stale = 0
    history: list[dict[str, float]] = []
    model = model.to(device)
    if manager.exists:
        saved = manager.load(device)
        if saved.get("complete") and best_path.exists():
            best = torch.load(best_path, map_location=device)
            model.load_state_dict(best["model_state"])
            return model, saved
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        epoch, seen = int(saved["epoch"]), int(saved["seen"])
        loss_sum, loss_count = float(saved["loss_sum"]), int(saved["loss_count"])
        best_key = tuple(saved["best_key"])
        stale = int(saved["stale"])
        history = saved["history"]
        restore_rng_state(saved.get("rng_state"))

    def checkpoint_payload(complete: bool = False) -> dict[str, Any]:
        return {
            "complete": complete,
            "run_name": run_name,
            "multitask": multitask,
            "epoch": epoch,
            "seen": seen,
            "loss_sum": loss_sum,
            "loss_count": loss_count,
            "best_key": best_key,
            "stale": stale,
            "history": history,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
        }

    while epoch < max_epochs:
        sampler = ResumableRandomSampler(dataset, int(config["seed"]), epoch, seen)
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["batch_size"]),
            sampler=sampler,
            **_loader_kwargs(config, device),
        )
        model.train()
        model.freeze_batchnorm_statistics()
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits_7, logits_2 = model(images)
            loss = criterion(logits_7, labels)
            if multitask:
                pair_mask = (labels == MEL_IDX) | (labels == NV_IDX)
                if pair_mask.any():
                    targets_2 = (labels[pair_mask] == NV_IDX).long()
                    auxiliary = nn.functional.cross_entropy(logits_2[pair_mask], targets_2)
                    loss = loss + float(cfg["auxiliary_loss_weight"]) * auxiliary
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head_parameters + backbone_parameters, float(cfg["gradient_clip_norm"]))
            optimizer.step()
            batch_count = len(labels)
            seen += batch_count
            loss_sum += float(loss.detach()) * batch_count
            loss_count += batch_count
            manager.maybe_save(checkpoint_payload)

        validation_key = _validation_score(model, val_frame, config, device)
        epoch_loss = loss_sum / max(1, loss_count)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": epoch_loss,
                "validation_macro_f1": validation_key[0],
                "validation_mel_sensitivity": validation_key[1],
            }
        )
        if validation_key > best_key:
            best_key = validation_key
            stale = 0
            atomic_torch_save(
                {"model_state": {key: value.cpu() for key, value in model.state_dict().items()}, "key": best_key},
                best_path,
            )
        else:
            stale += 1
        scheduler.step()
        epoch += 1
        seen, loss_sum, loss_count = 0, 0.0, 0
        manager.save(checkpoint_payload(False))
        if epoch >= min_epochs and stale >= patience:
            break
    manager.save(checkpoint_payload(True))
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    with (output_dir / f"{run_name}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return model, checkpoint_payload(True)


def benchmark_training_steps(
    model: HardPairNet,
    train_frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    steps: int = 100,
    warmup: int = 10,
) -> dict[str, float | str]:
    dataset = HAM10000Dataset(
        train_frame, build_transform(int(config["dataset"]["image_size"]), training=True)
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, **_loader_kwargs(config, device))
    model = model.to(device)
    model.freeze_for_partial_finetuning(int(config["model"]["unfreeze_last_modules"]))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    iterator = iter(loader)
    for step in range(warmup + steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits, _ = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= warmup:
            timings.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    return {
        "device": str(device),
        "steps": float(steps),
        "seconds_per_step": float(np.mean(timings)),
        "peak_allocated_gb": float(peak),
    }

