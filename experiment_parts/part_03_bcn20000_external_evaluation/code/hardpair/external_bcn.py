"""Frozen, resumable BCN20000 external evaluation for HardPair-Net.

This module deliberately contains no fitting operation. It verifies hashes of
all frozen artifacts, caches deterministic inference in small atomic chunks,
and evaluates only after inference has completed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from .config import seed_everything
from .data import CLASS_NAMES, CLASS_TO_IDX, MEL_IDX, NV_IDX, build_transform, seed_worker
from .evaluation import expected_calibration_error
from .fusion import apply_gate, pair_restricted_fusion
from .model import HardPairNet, LogisticGate


METHODS = (
    "baseline",
    "auxiliary_only",
    "original_fixed_fusion",
    "original_learned_fusion",
    "balanced_logistic_fixed_fusion",
    "balanced_logistic_learned_gate",
)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_external_config(path: str | Path = "bcn_external_evaluation/external_evaluation_config.yaml") -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    for key in ("metadata_csv", "image_dir"):
        config["data"][key] = str(_resolve(path.parent, config["data"][key]))
    for spec in config["frozen_artifacts"].values():
        if isinstance(spec, dict) and "path" in spec:
            spec["path"] = str(_resolve(path.parent, spec["path"]))
    audit = config["frozen_artifacts"].get("audit_facts")
    if isinstance(audit, str):
        config["frozen_artifacts"]["audit_facts"] = str(_resolve(path.parent, audit))
    return config


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _manifest_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["image_id", "lesion_id", "external_diagnosis", "label"]
    values = frame[columns].astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def build_bcn_manifest(config: dict[str, Any], verify_images: bool = False) -> pd.DataFrame:
    data = config["data"]
    metadata_path = Path(data["metadata_csv"])
    expected_hash = str(data["metadata_sha256"]).upper()
    actual_hash = sha256_file(metadata_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"BCN metadata hash changed: expected {expected_hash}, got {actual_hash}")
    metadata = pd.read_csv(metadata_path)
    required = {data["filename_column"], data["diagnosis_column"], data["lesion_column"]}
    missing_columns = required.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"BCN metadata is missing columns: {sorted(missing_columns)}")
    if len(metadata) != int(data["expected_rows"]):
        raise ValueError(f"Expected {data['expected_rows']} BCN rows, found {len(metadata)}")
    filename_column = data["filename_column"]
    diagnosis_column = data["diagnosis_column"]
    lesion_column = data["lesion_column"]
    if metadata[filename_column].duplicated().any():
        raise ValueError("BCN metadata contains duplicate filenames")
    image_dir = Path(data["image_dir"])
    metadata = metadata.copy()
    metadata["image_id"] = metadata[filename_column].map(lambda value: Path(str(value)).stem)
    metadata["image_path"] = metadata[filename_column].map(lambda value: str((image_dir / str(value)).resolve()))
    metadata["lesion_id"] = metadata[lesion_column].astype(str)
    metadata["external_diagnosis"] = metadata[diagnosis_column].astype(str).str.upper()
    mapped = metadata["external_diagnosis"].map(data["label_map"])
    metadata["dx"] = mapped.fillna("excluded")
    metadata["label"] = mapped.map(CLASS_TO_IDX).fillna(-1).astype(int)
    missing_files = [path for path in metadata["image_path"] if not Path(path).is_file()]
    disk_files = {path.name for path in image_dir.iterdir() if path.is_file()}
    expected_files = set(metadata[filename_column].astype(str))
    extra_files = sorted(disk_files - expected_files)
    if missing_files or extra_files:
        raise FileNotFoundError(
            f"BCN image mismatch: {len(missing_files)} missing and {len(extra_files)} extra files"
        )
    if metadata.groupby("lesion_id")["external_diagnosis"].nunique().max() != 1:
        raise ValueError("At least one BCN lesion has conflicting diagnoses")
    if verify_images:
        failures: list[str] = []
        for path in metadata["image_path"]:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                failures.append(f"{path}: {exc}")
        if failures:
            raise ValueError(f"Unreadable BCN images ({len(failures)}): {failures[:3]}")
    return metadata.reset_index(drop=True)


def verify_frozen_artifacts(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, spec in config["frozen_artifacts"].items():
        if not isinstance(spec, dict) or "sha256" not in spec:
            continue
        path = Path(spec["path"])
        actual = sha256_file(path)
        expected = str(spec["sha256"]).upper()
        rows.append({"artifact": name, "path": str(path), "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected})
    result = pd.DataFrame(rows)
    if not result["matches"].all():
        bad = result.loc[~result["matches"], "artifact"].tolist()
        raise RuntimeError(f"Frozen artifact hash mismatch: {bad}")
    audit_path = Path(config["frozen_artifacts"]["audit_facts"])
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    for name, value in (
        ("baseline_generalist", audit["temperatures"]["baseline_generalist"]),
        ("multitask_generalist", audit["temperatures"]["multitask_generalist"]),
        ("original_specialist", audit["temperatures"]["specialist"]),
    ):
        if not math.isclose(float(value), float(config["temperatures"][name]), rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"Frozen temperature mismatch for {name}")
    specialist = joblib.load(config["frozen_artifacts"]["balanced_logistic_specialist"]["path"])
    if not math.isclose(
        float(specialist["temperature"]),
        float(config["temperatures"]["balanced_logistic_specialist"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError("Frozen balanced-logistic specialist temperature mismatch")
    return result


def preflight_external_cohort(
    config: dict[str, Any], output_dir: str | Path, verify_images: bool = True
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_bcn_manifest(config, verify_images=verify_images)
    artifact_audit = verify_frozen_artifacts(config)
    artifact_audit.to_csv(output_dir / "frozen_artifact_audit.csv", index=False)
    image_counts = manifest["external_diagnosis"].value_counts().sort_index()
    lesion_counts = manifest.groupby("external_diagnosis")["lesion_id"].nunique().sort_index()
    summary = pd.DataFrame({"diagnosis": image_counts.index, "images": image_counts.values, "lesions": lesion_counts.reindex(image_counts.index).values})
    summary.to_csv(output_dir / "cohort_summary.csv", index=False)
    manifest.to_csv(output_dir / "bcn_manifest.csv", index=False)
    _atomic_json(
        {
            "rows": int(len(manifest)),
            "lesions": int(manifest["lesion_id"].nunique()),
            "primary_rows_before_overlap_exclusion": int((manifest["label"] >= 0).sum()),
            "excluded_scc_or_unknown": int((manifest["label"] < 0).sum()),
            "manifest_fingerprint": _manifest_fingerprint(manifest),
            "images_verified": bool(verify_images),
        },
        output_dir / "preflight.json",
    )
    return manifest


def stratified_smoke_manifest(frame: pd.DataFrame, count: int, seed: int = 42) -> pd.DataFrame:
    if count <= 0 or count >= len(frame):
        return frame.copy().reset_index(drop=True)
    diagnoses = frame["external_diagnosis"].unique()
    if count < len(diagnoses):
        raise ValueError(f"Smoke count must be at least {len(diagnoses)} to retain every diagnosis")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    remaining: list[int] = []
    for _, group in frame.groupby("external_diagnosis", sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        chosen.append(int(indices[0]))
        remaining.extend(int(value) for value in indices[1:])
    rng.shuffle(remaining)
    chosen.extend(remaining[: count - len(chosen)])
    return frame.loc[chosen].sort_values(["external_diagnosis", "image_id"]).reset_index(drop=True)


def _phash64(path: str | Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    positions = np.arange(32)
    basis = np.cos((2 * positions[None, :] + 1) * positions[:, None] * np.pi / 64.0)
    basis[0] *= 1.0 / np.sqrt(2.0)
    coefficients = (2.0 / 32.0) * basis @ pixels @ basis.T
    low = coefficients[:8, :8].reshape(-1)
    threshold = np.median(low[1:])
    result = 0
    for bit in low > threshold:
        result = (result << 1) | int(bit)
    return result


class _BKTree:
    def __init__(self) -> None:
        self.root: list[Any] | None = None

    def add(self, value: int, payload: str) -> None:
        if self.root is None:
            self.root = [value, [payload], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(payload)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, [payload], {}]
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        results: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                results.extend((distance, payload) for payload in node[1])
            lower, upper = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if lower <= edge <= upper)
        return results


def _load_ham_manifest(root: Path) -> pd.DataFrame:
    split_path = root / "outputs" / "ham10000_splits.csv"
    frame = pd.read_csv(split_path)
    if "image_path" not in frame or not frame["image_path"].map(lambda value: Path(str(value)).is_file()).all():
        lookup = {}
        for directory in (root / "HAM10000_images_part_1", root / "HAM10000_images_part_2"):
            lookup.update({path.stem: str(path.resolve()) for path in directory.glob("*.*")})
        frame["image_path"] = frame["image_id"].map(lookup)
    if frame["image_path"].isna().any():
        raise FileNotFoundError("Could not resolve all HAM10000 images for overlap audit")
    return frame


def _hash_frame(frame: pd.DataFrame, dataset: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"sha256": str, "phash64": str})
        completed = set(cached["image_id"].astype(str))
    else:
        cached = pd.DataFrame(columns=["dataset", "image_id", "image_path", "sha256", "phash64"])
        completed = set()
    pending: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        image_id = str(row.image_id)
        if image_id in completed:
            continue
        pending.append(
            {
                "dataset": dataset,
                "image_id": image_id,
                "image_path": str(row.image_path),
                "sha256": sha256_file(row.image_path),
                "phash64": f"{_phash64(row.image_path):016X}",
            }
        )
        if len(pending) >= 250:
            cached = pd.concat([cached, pd.DataFrame(pending)], ignore_index=True)
            temporary = cache_path.with_suffix(".tmp.csv")
            cached.to_csv(temporary, index=False)
            temporary.replace(cache_path)
            pending.clear()
            print(f"{dataset} overlap hashes: {len(cached):,}/{len(frame):,}")
    if pending:
        cached = pd.concat([cached, pd.DataFrame(pending)], ignore_index=True)
        temporary = cache_path.with_suffix(".tmp.csv")
        cached.to_csv(temporary, index=False)
        temporary.replace(cache_path)
        print(f"{dataset} overlap hashes: {len(cached):,}/{len(frame):,}")
    expected = set(frame["image_id"].astype(str))
    result = cached.loc[cached["image_id"].astype(str).isin(expected)].copy()
    if len(result) != len(frame) or result["image_id"].duplicated().any():
        raise RuntimeError(f"Hash cache alignment failure for {dataset}")
    return result


def audit_ham_overlap(
    bcn_manifest: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    near_duplicate_radius: int = 4,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    root = Path(config["_config_dir"]).parent
    ham = _load_ham_manifest(root)
    cache_dir = output_dir / "hash_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ham_hashes = _hash_frame(ham, "HAM10000", cache_dir / "ham_hashes.csv")
    bcn_hashes = _hash_frame(bcn_manifest, "BCN20000", cache_dir / "bcn_hashes.csv")
    exact_lookup: dict[str, list[str]] = defaultdict(list)
    tree = _BKTree()
    for row in ham_hashes.itertuples(index=False):
        exact_lookup[str(row.sha256)].append(str(row.image_id))
        tree.add(int(str(row.phash64), 16), str(row.image_id))
    rows: list[dict[str, Any]] = []
    for row in bcn_hashes.itertuples(index=False):
        exact = exact_lookup.get(str(row.sha256), [])
        candidates = sorted(tree.query(int(str(row.phash64), 16), near_duplicate_radius))
        if exact or candidates:
            for distance, ham_id in candidates or [(0, value) for value in exact]:
                rows.append(
                    {
                        "bcn_image_id": str(row.image_id),
                        "ham_image_id": ham_id,
                        "exact_sha256_match": ham_id in exact,
                        "phash_hamming_distance": int(distance),
                        "near_duplicate_candidate": int(distance) <= near_duplicate_radius,
                    }
                )
    columns = ["bcn_image_id", "ham_image_id", "exact_sha256_match", "phash_hamming_distance", "near_duplicate_candidate"]
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(output_dir / "overlap_candidates.csv", index=False)
    _atomic_json(
        {
            "ham_images_hashed": int(len(ham_hashes)),
            "bcn_images_hashed": int(len(bcn_hashes)),
            "exact_cross_dataset_pairs": int(result["exact_sha256_match"].sum()) if len(result) else 0,
            "exact_bcn_images": int(result.loc[result["exact_sha256_match"], "bcn_image_id"].nunique()) if len(result) else 0,
            "near_duplicate_radius": int(near_duplicate_radius),
            "near_duplicate_candidate_pairs": int(result["near_duplicate_candidate"].sum()) if len(result) else 0,
            "near_duplicate_candidate_bcn_images": int(result.loc[result["near_duplicate_candidate"], "bcn_image_id"].nunique()) if len(result) else 0,
            "note": "Perceptual-hash hits are screening candidates, not proven duplicate lesions.",
        },
        output_dir / "overlap_summary.json",
    )
    return result


class _ExternalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transform(image_size, training=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "index": index, "image_id": row.image_id}


def _load_frozen_models(config: dict[str, Any], device: torch.device) -> tuple[HardPairNet, HardPairNet, HardPairNet]:
    artifacts = config["frozen_artifacts"]
    baseline = HardPairNet(pretrained=False)
    baseline.load_state_dict(torch.load(artifacts["baseline_checkpoint"]["path"], map_location="cpu")["model_state"])
    multitask = HardPairNet(pretrained=False)
    multitask.load_state_dict(torch.load(artifacts["multitask_checkpoint"]["path"], map_location="cpu")["model_state"])
    feature_model = HardPairNet(pretrained=True)
    return baseline.to(device).eval(), multitask.to(device).eval(), feature_model.to(device).eval()


def _chunk_paths(chunk_dir: Path, count: int, chunk_size: int) -> list[Path]:
    return [chunk_dir / f"chunk_{start:06d}_{min(count, start + chunk_size):06d}.pt" for start in range(0, count, chunk_size)]


def _collate_prediction_chunks(paths: Iterable[Path], expected_fingerprint: str) -> dict[str, Any]:
    chunks = [torch.load(path, map_location="cpu") for path in paths]
    if not chunks:
        raise RuntimeError("No inference chunks were produced")
    if any(chunk["manifest_fingerprint"] != expected_fingerprint for chunk in chunks):
        raise RuntimeError("Inference cache belongs to a different manifest")
    starts = [int(chunk["start"]) for chunk in chunks]
    if starts != sorted(starts):
        chunks.sort(key=lambda value: int(value["start"]))
    return {
        "manifest_fingerprint": expected_fingerprint,
        "image_ids": [value for chunk in chunks for value in chunk["image_ids"]],
        "baseline_logits": torch.cat([chunk["baseline_logits"] for chunk in chunks]),
        "multitask_logits": torch.cat([chunk["multitask_logits"] for chunk in chunks]),
        "original_specialist_logits": torch.cat([chunk["original_specialist_logits"] for chunk in chunks]),
        "pretrained_features": torch.cat([chunk["pretrained_features"] for chunk in chunks]),
    }


def cache_external_predictions(
    manifest: pd.DataFrame, config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    inference = config["inference"]
    chunk_size = int(inference["chunk_size"])
    chunk_dir = output_dir / "inference_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _manifest_fingerprint(manifest)
    paths = _chunk_paths(chunk_dir, len(manifest), chunk_size)
    complete = True
    for path in paths:
        if not path.exists():
            complete = False
            break
        try:
            chunk = torch.load(path, map_location="cpu")
            if chunk.get("manifest_fingerprint") != fingerprint:
                raise RuntimeError(f"Stale inference chunk: {path}")
        except Exception:
            complete = False
            break
    if complete:
        return _collate_prediction_chunks(paths, fingerprint)

    requested = str(inference["device"]).lower()
    device = torch.device(requested if not requested.startswith("cuda") or torch.cuda.is_available() else "cpu")
    seed_everything(int(config["protocol"]["seed"]), deterministic=bool(inference["deterministic"]))
    verify_frozen_artifacts(config)
    baseline, multitask, feature_model = _load_frozen_models(config, device)
    started = time.perf_counter()
    for start, path in zip(range(0, len(manifest), chunk_size), paths):
        stop = min(len(manifest), start + chunk_size)
        if path.exists():
            chunk = torch.load(path, map_location="cpu")
            if chunk.get("manifest_fingerprint") == fingerprint and int(chunk.get("start", -1)) == start and int(chunk.get("stop", -1)) == stop:
                continue
            raise RuntimeError(f"Stale or invalid inference chunk: {path}")
        subset = manifest.iloc[start:stop].reset_index(drop=True)
        dataset = _ExternalDataset(subset, int(config["data"]["image_size"]))
        generator = torch.Generator().manual_seed(int(config["protocol"]["seed"]) + start)
        loader = DataLoader(
            dataset,
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=int(inference["num_workers"]),
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=int(inference["num_workers"]) > 0,
        )
        baseline_logits: list[torch.Tensor] = []
        multitask_logits: list[torch.Tensor] = []
        specialist_logits: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                base7, _ = baseline(images)
                multi7, multi2 = multitask(images)
                raw_features = feature_model.extract_features(images)
                baseline_logits.append(base7.float().cpu().clone())
                multitask_logits.append(multi7.float().cpu().clone())
                specialist_logits.append(multi2.float().cpu().clone())
                features.append(raw_features.float().cpu().clone())
        payload = {
            "manifest_fingerprint": fingerprint,
            "start": start,
            "stop": stop,
            "image_ids": subset["image_id"].tolist(),
            "baseline_logits": torch.cat(baseline_logits),
            "multitask_logits": torch.cat(multitask_logits),
            "original_specialist_logits": torch.cat(specialist_logits),
            "pretrained_features": torch.cat(features),
        }
        _atomic_torch_save(payload, path)
        _atomic_json(
            {
                "manifest_fingerprint": fingerprint,
                "completed_images": stop,
                "total_images": len(manifest),
                "last_checkpoint": str(path),
                "checkpoints_are_more_frequent_than_hourly": True,
                "elapsed_seconds_this_session": time.perf_counter() - started,
                "device": str(device),
            },
            output_dir / "inference_progress.json",
        )
        print(f"External inference: {stop:,}/{len(manifest):,} images cached")
    cache = _collate_prediction_chunks(paths, fingerprint)
    if cache["image_ids"] != manifest["image_id"].tolist():
        raise RuntimeError("External inference cache is not aligned to the manifest")
    _atomic_torch_save(cache, output_dir / "external_prediction_cache.pt")
    return cache


def _two_class_logits(classifier: Any, scaled_features: np.ndarray) -> torch.Tensor:
    score = classifier.decision_function(scaled_features)
    return torch.tensor(np.column_stack([-0.5 * score, 0.5 * score]), dtype=torch.float32)


def build_external_probabilities(cache: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    temperature = config["temperatures"]
    baseline = torch.softmax(cache["baseline_logits"] / float(temperature["baseline_generalist"]), dim=1)
    auxiliary = torch.softmax(cache["multitask_logits"] / float(temperature["multitask_generalist"]), dim=1)
    original_specialist = torch.softmax(
        cache["original_specialist_logits"] / float(temperature["original_specialist"]), dim=1
    )
    specialist_artifact = joblib.load(config["frozen_artifacts"]["balanced_logistic_specialist"]["path"])
    scaled = specialist_artifact["scaler"].transform(cache["pretrained_features"].numpy())
    balanced_logits = _two_class_logits(specialist_artifact["classifier"], scaled)
    balanced_specialist = torch.softmax(
        balanced_logits / float(temperature["balanced_logistic_specialist"]), dim=1
    )
    original_gate_artifact = torch.load(config["frozen_artifacts"]["original_gate"]["path"], map_location="cpu")
    original_gate = LogisticGate(max_alpha=0.75, initial_alpha=0.5)
    original_gate.load_state_dict(original_gate_artifact["gate_state"])
    original_learned, original_alpha = apply_gate(
        original_gate,
        {"mean": original_gate_artifact["mean"], "std": original_gate_artifact["std"]},
        auxiliary,
        original_specialist,
    )
    balanced_gate_artifact = torch.load(config["frozen_artifacts"]["balanced_logistic_gate"]["path"], map_location="cpu")
    balanced_gate = LogisticGate(max_alpha=0.75, initial_alpha=0.5)
    balanced_gate.load_state_dict(balanced_gate_artifact["gate_state"])
    balanced_learned, balanced_alpha = apply_gate(
        balanced_gate,
        {"mean": balanced_gate_artifact["mean"], "std": balanced_gate_artifact["std"]},
        auxiliary,
        balanced_specialist,
    )
    probabilities = {
        "baseline": baseline,
        "auxiliary_only": auxiliary,
        "original_fixed_fusion": pair_restricted_fusion(auxiliary, original_specialist, float(config["methods"]["fixed_alpha"])),
        "original_learned_fusion": original_learned,
        "balanced_logistic_fixed_fusion": pair_restricted_fusion(auxiliary, balanced_specialist, float(config["methods"]["fixed_alpha"])),
        "balanced_logistic_learned_gate": balanced_learned,
    }
    extras = {
        "original_specialist": original_specialist,
        "balanced_logistic_specialist": balanced_specialist,
        "original_alpha": original_alpha,
        "balanced_alpha": balanced_alpha,
    }
    return probabilities, extras


def _metric_row(name: str, p: np.ndarray, y: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    prediction = p.argmax(1)
    recalls = {}
    for index, class_name in enumerate(CLASS_NAMES):
        mask = y == index
        recalls[class_name] = float((prediction[mask] == index).mean()) if mask.any() else 0.0
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[y]
    offpair = ~np.isin(y, [MEL_IDX, NV_IDX])
    reference_prediction = reference.argmax(1)
    offpair_changed = offpair & (prediction != reference_prediction)
    offpair_to_pair = offpair & ~np.isin(reference_prediction, [MEL_IDX, NV_IDX]) & np.isin(prediction, [MEL_IDX, NV_IDX])
    pair_mass = p[:, MEL_IDX] + p[:, NV_IDX]
    strongest_other = np.delete(p, [MEL_IDX, NV_IDX], axis=1).max(1)
    mel_mask = y == MEL_IDX
    offpair_indices = [index for index in range(7) if index not in (MEL_IDX, NV_IDX)]
    preservation = float(np.abs(p[:, offpair_indices] - reference[:, offpair_indices]).max()) if len(p) else 0.0
    return {
        "model": name,
        "images": int(len(y)),
        "lesions": None,
        "pair_balanced_accuracy": float((recalls["mel"] + recalls["nv"]) / 2.0),
        "mel_sensitivity": recalls["mel"],
        "nv_sensitivity": recalls["nv"],
        "mel_to_nv": int(((y == MEL_IDX) & (prediction == NV_IDX)).sum()),
        "nv_to_mel": int(((y == NV_IDX) & (prediction == MEL_IDX)).sum()),
        "macro_f1": float(f1_score(y, prediction, labels=range(7), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "mel_pr_auc": float(average_precision_score((y == MEL_IDX).astype(int), p[:, MEL_IDX])),
        "brier_score": float(np.square(p - one_hot).sum(1).mean()),
        "ece_15": float(expected_calibration_error(p, y, bins=15)),
        "unrecoverable_melanoma_count": int((mel_mask & (pair_mass <= strongest_other)).sum()),
        "unrecoverable_melanoma_rate": float((mel_mask & (pair_mass <= strongest_other)).sum() / max(1, mel_mask.sum())),
        "offpair_prediction_changes": int(offpair_changed.sum()),
        "offpair_prediction_change_rate": float(offpair_changed.sum() / max(1, offpair.sum())),
        "offpair_to_pair_transitions": int(offpair_to_pair.sum()),
        "offpair_to_pair_transition_rate": float(offpair_to_pair.sum() / max(1, offpair.sum())),
        "max_offpair_probability_difference": preservation,
        "max_probability_sum_error": float(np.abs(p.sum(1) - 1.0).max()),
    }


def _confusion_metrics(matrix: np.ndarray, metric: str) -> float:
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    diagonal = np.diag(matrix)
    recalls = np.divide(diagonal, support, out=np.zeros(7), where=support > 0)
    if metric == "pair_balanced_accuracy":
        return float((recalls[MEL_IDX] + recalls[NV_IDX]) / 2.0)
    if metric == "mel_sensitivity":
        return float(recalls[MEL_IDX])
    if metric == "nv_sensitivity":
        return float(recalls[NV_IDX])
    if metric == "macro_f1":
        denominator = support + predicted
        f1 = np.divide(2.0 * diagonal, denominator, out=np.zeros(7), where=denominator > 0)
        return float(f1.mean())
    raise KeyError(metric)


def _group_confusions(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(groups, return_inverse=True)
    matrices = np.zeros((len(unique), 7, 7), dtype=np.int64)
    np.add.at(matrices, (inverse, y, prediction), 1)
    return unique, matrices


def paired_lesion_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    lesion_ids: np.ndarray,
    metrics: Iterable[str],
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    unique_c, candidate_matrices = _group_confusions(labels, candidate.argmax(1), lesion_ids)
    unique_r, reference_matrices = _group_confusions(labels, reference.argmax(1), lesion_ids)
    if not np.array_equal(unique_c, unique_r):
        raise RuntimeError("Group alignment failure during bootstrap")
    rng = np.random.default_rng(seed)
    differences = {metric: np.empty(resamples, dtype=np.float64) for metric in metrics}
    for index in range(resamples):
        weights = rng.multinomial(len(unique_c), np.full(len(unique_c), 1.0 / len(unique_c)))
        candidate_matrix = np.tensordot(weights, candidate_matrices, axes=(0, 0))
        reference_matrix = np.tensordot(weights, reference_matrices, axes=(0, 0))
        for metric in metrics:
            differences[metric][index] = _confusion_metrics(candidate_matrix, metric) - _confusion_metrics(reference_matrix, metric)
    candidate_total = candidate_matrices.sum(0)
    reference_total = reference_matrices.sum(0)
    rows = []
    for metric in metrics:
        distribution = differences[metric]
        estimate = _confusion_metrics(candidate_total, metric) - _confusion_metrics(reference_total, metric)
        low, high = np.percentile(distribution, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "difference": float(estimate),
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "bootstrap_mean_difference": float(distribution.mean()),
                "resamples": int(resamples),
                "lesions": int(len(unique_c)),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return rows


def _overlap_flags(manifest: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = output_dir / "overlap_candidates.csv"
    if not path.exists():
        raise FileNotFoundError("Run the overlap audit before evaluation")
    overlap = pd.read_csv(path)
    if overlap.empty:
        return np.zeros(len(manifest), dtype=bool), np.zeros(len(manifest), dtype=bool)
    exact_ids = set(overlap.loc[overlap["exact_sha256_match"].astype(bool), "bcn_image_id"].astype(str))
    near_ids = set(overlap.loc[overlap["near_duplicate_candidate"].astype(bool), "bcn_image_id"].astype(str))
    ids = manifest["image_id"].astype(str)
    return ids.isin(exact_ids).to_numpy(), ids.isin(near_ids).to_numpy()


def evaluate_external_predictions(
    manifest: pd.DataFrame,
    cache: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    if cache["image_ids"] != manifest["image_id"].tolist():
        raise RuntimeError("Prediction cache and BCN manifest are misaligned")
    verify_frozen_artifacts(config)
    probabilities, extras = build_external_probabilities(cache, config)
    exact_overlap, near_overlap = _overlap_flags(manifest, output_dir)
    labels = manifest["label"].to_numpy(dtype=int)
    primary_mask = (labels >= 0) & ~exact_overlap
    if not primary_mask.any():
        raise RuntimeError("Primary external cohort is empty")
    y = labels[primary_mask]
    lesions = manifest.loc[primary_mask, "lesion_id"].astype(str).to_numpy()
    auxiliary_reference = probabilities["auxiliary_only"][primary_mask].numpy()
    metric_rows = []
    matrices: dict[str, Any] = {}
    for name in METHODS:
        values = probabilities[name][primary_mask].numpy()
        reference = values if name in ("baseline", "auxiliary_only") else auxiliary_reference
        row = _metric_row(name, values, y, reference)
        row["lesions"] = int(np.unique(lesions).size)
        metric_rows.append(row)
        matrices[name] = confusion_matrix(y, values.argmax(1), labels=range(7)).tolist()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "external_metrics.csv", index=False)
    _atomic_json({"class_order": CLASS_NAMES, "matrices": matrices}, output_dir / "external_confusion_matrices.json")

    prediction_frame = manifest[["image_id", "lesion_id", "external_diagnosis", "dx", "label"]].copy()
    prediction_frame["excluded_from_primary"] = ~primary_mask
    prediction_frame["exact_ham_overlap"] = exact_overlap
    prediction_frame["near_duplicate_candidate"] = near_overlap
    for name, values in probabilities.items():
        array = values.numpy()
        prediction_frame[f"{name}_prediction_index"] = array.argmax(1)
        prediction_frame[f"{name}_prediction"] = [CLASS_NAMES[index] for index in array.argmax(1)]
        for index, class_name in enumerate(CLASS_NAMES):
            prediction_frame[f"{name}_p_{class_name}"] = array[:, index]
    prediction_frame["original_gate_alpha"] = extras["original_alpha"].numpy().reshape(-1)
    prediction_frame["balanced_logistic_gate_alpha"] = extras["balanced_alpha"].numpy().reshape(-1)
    prediction_frame.to_csv(output_dir / "external_image_predictions.csv", index=False)

    scc = prediction_frame.loc[prediction_frame["external_diagnosis"] == "SCC"]
    scc_rows = []
    for name in METHODS:
        counts = scc[f"{name}_prediction"].value_counts()
        for class_name in CLASS_NAMES:
            count = int(counts.get(class_name, 0))
            scc_rows.append({"model": name, "forced_prediction": class_name, "count": count, "rate": count / max(1, len(scc)), "scc_images": int(len(scc))})
    pd.DataFrame(scc_rows).to_csv(output_dir / "scc_forced_prediction_distribution.csv", index=False)

    bootstrap_rows: list[dict[str, Any]] = []
    resamples = int(config["statistics"]["bootstrap_resamples"])
    bootstrap_metrics = config["statistics"]["primary_metrics"]
    for comparison_index, comparison in enumerate(config["statistics"]["predeclared_comparisons"]):
        candidate_name = comparison["candidate"]
        reference_name = comparison["reference"]
        rows = paired_lesion_bootstrap(
            probabilities[candidate_name][primary_mask].numpy(),
            probabilities[reference_name][primary_mask].numpy(),
            y,
            lesions,
            bootstrap_metrics,
            resamples,
            int(config["protocol"]["seed"]) + comparison_index,
        )
        for row in rows:
            row.update({"candidate": candidate_name, "reference": reference_name, "role": comparison["role"]})
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_dir / "external_bootstrap.csv", index=False)

    sensitivity_mask = primary_mask & ~near_overlap
    sensitivity_rows = []
    for name in METHODS:
        values = probabilities[name][sensitivity_mask].numpy()
        reference = values if name in ("baseline", "auxiliary_only") else probabilities["auxiliary_only"][sensitivity_mask].numpy()
        row = _metric_row(name, values, labels[sensitivity_mask], reference)
        row["lesions"] = int(manifest.loc[sensitivity_mask, "lesion_id"].nunique())
        row["excluded_near_duplicate_candidates"] = int((primary_mask & near_overlap).sum())
        sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "near_duplicate_sensitivity_metrics.csv", index=False)

    fixed_name = str(config["methods"]["primary_candidate"])
    primary_comparison = bootstrap.loc[
        (bootstrap["candidate"] == fixed_name) & (bootstrap["reference"] == "baseline")
    ]
    metric_lookup = metrics.set_index("model")
    lines = [
        "# BCN20000 external evaluation",
        "",
        "This is a frozen external confirmation. BCN labels were not used to fit or select any model, temperature, gate, threshold, or fusion coefficient.",
        "",
        "## Cohort",
        "",
        f"- Local labeled release: {len(manifest):,} images from {manifest['lesion_id'].nunique():,} lesions.",
        f"- Primary seven-class cohort: {int(primary_mask.sum()):,} images from {np.unique(lesions).size:,} lesions.",
        f"- SCC/unknown excluded from primary analysis: {int((labels < 0).sum()):,} images.",
        f"- Exact HAM10000 overlaps excluded: {int(exact_overlap.sum()):,} images.",
        f"- Perceptual-hash candidates (screening only): {int(near_overlap.sum()):,} images.",
        "",
        "## Predeclared primary result",
        "",
        f"The exploratory balanced-logistic fixed fusion (alpha=0.5) achieved pair-balanced accuracy {metric_lookup.loc[fixed_name, 'pair_balanced_accuracy']:.4f}, compared with {metric_lookup.loc['baseline', 'pair_balanced_accuracy']:.4f} for the frozen baseline.",
        "",
        "Paired lesion-level differences (candidate minus baseline):",
        "",
    ]
    for row in primary_comparison.itertuples(index=False):
        lines.append(f"- {row.metric}: {row.difference:+.4f} (95% CI {row.ci_95_low:+.4f} to {row.ci_95_high:+.4f}).")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "SCC is outside the frozen seven-class label space, so its outputs are forced predictions and are not accuracy estimates. Perceptual-hash matches are candidates for manual review, not proof of shared lesions. Statistical intervals quantify sampling uncertainty for this cohort and do not remove dataset-shift limitations.",
        ]
    )
    (output_dir / "external_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(
        {
            "primary_images": int(primary_mask.sum()),
            "primary_lesions": int(np.unique(lesions).size),
            "excluded_scc_or_unknown": int((labels < 0).sum()),
            "excluded_exact_overlap": int(exact_overlap.sum()),
            "near_duplicate_candidates": int(near_overlap.sum()),
            "models": list(METHODS),
            "bootstrap_resamples": resamples,
            "no_external_fitting_performed": True,
        },
        output_dir / "evaluation_summary.json",
    )
    return metrics, bootstrap

