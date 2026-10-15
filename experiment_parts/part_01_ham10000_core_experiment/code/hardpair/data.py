from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
MEL_IDX = CLASS_TO_IDX["mel"]
NV_IDX = CLASS_TO_IDX["nv"]


def build_manifest(config: dict[str, Any], verify_images: bool = False) -> pd.DataFrame:
    metadata = pd.read_csv(config["paths"]["metadata_csv"])
    required = {"image_id", "lesion_id", "dx"}
    missing_columns = required.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate image_id values")
    unknown = sorted(set(metadata["dx"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown diagnoses: {unknown}")

    image_lookup: dict[str, str] = {}
    duplicate_files: list[str] = []
    for directory in config["paths"]["image_dirs"]:
        for path in Path(directory).glob("*.*"):
            if path.stem in image_lookup:
                duplicate_files.append(path.stem)
            image_lookup[path.stem] = str(path.resolve())
    if duplicate_files:
        raise ValueError(f"Duplicate image files found for: {duplicate_files[:5]}")
    metadata["image_path"] = metadata["image_id"].map(image_lookup)
    missing = metadata.loc[metadata["image_path"].isna(), "image_id"].tolist()
    if missing:
        raise FileNotFoundError(f"No image file for {len(missing)} IDs, e.g. {missing[:5]}")
    metadata["label"] = metadata["dx"].map(CLASS_TO_IDX).astype(int)

    if verify_images:
        errors: list[str] = []
        for image_path in metadata["image_path"]:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # pragma: no cover - depends on corrupt inputs
                errors.append(f"{image_path}: {exc}")
        if errors:
            raise ValueError(f"Unreadable images ({len(errors)}): {errors[:3]}")
    return metadata


def create_split_manifest(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    seed = int(config["seed"])
    group_column = config["dataset"]["grouping_column"]
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    result = manifest.copy()
    result["fold"] = -1
    for fold, (_, held_out) in enumerate(
        splitter.split(result, y=result["label"], groups=result[group_column])
    ):
        result.iloc[held_out, result.columns.get_loc("fold")] = fold
    fold_map = {
        int(config["dataset"]["fusion_fold"]): "fusion",
        int(config["dataset"]["validation_fold"]): "val",
        int(config["dataset"]["test_fold"]): "test",
    }
    result["split"] = result["fold"].map(fold_map).fillna("train")
    leakage = result.groupby(group_column)["split"].nunique()
    if int(leakage.max()) != 1:
        raise AssertionError("A lesion group appears in more than one split")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def stratified_smoke_sample(manifest: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    """Deterministic <=max_images sample retaining every available split/class stratum."""
    if len(manifest) <= max_images:
        return manifest.copy().reset_index(drop=True)
    if max_images < manifest.groupby(["split", "label"]).ngroups:
        raise ValueError("max_images is too small to retain every split/class stratum")
    rng = np.random.default_rng(seed)
    strata = list(manifest.groupby(["split", "label"], sort=True).groups.items())
    chosen: list[int] = []
    pools: dict[tuple[str, int], list[int]] = {}
    for key, indices in strata:
        shuffled = np.array(list(indices), dtype=int)
        rng.shuffle(shuffled)
        chosen.append(int(shuffled[0]))
        pools[key] = shuffled[1:].tolist()
    candidates = [index for pool in pools.values() for index in pool]
    rng.shuffle(candidates)
    chosen.extend(candidates[: max_images - len(chosen)])
    return manifest.loc[chosen].sort_values(["split", "label", "image_id"]).reset_index(drop=True)


class ResizePad:
    def __init__(self, size: int, fill: tuple[int, int, int] = (128, 128, 128)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return ImageOps.pad(resized, (self.size, self.size), color=self.fill, centering=(0.5, 0.5))


def build_transform(image_size: int, training: bool) -> transforms.Compose:
    operations: list[Any] = [ResizePad(image_size)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15, fill=(128, 128, 128)),
                transforms.ColorJitter(brightness=0.05, contrast=0.10),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": int(row.label),
            "index": index,
            "image_id": row.image_id,
            "lesion_id": row.lesion_id,
        }


class ResumableRandomSampler(Sampler[int]):
    """Epoch-deterministic shuffling with a sample offset for mid-epoch resume."""

    def __init__(self, data_source: Sequence[Any], seed: int, epoch: int, start_index: int = 0):
        self.data_source = data_source
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        self.indices = torch.randperm(len(data_source), generator=generator).tolist()[start_index:]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(manifest["split"], manifest["dx"], margins=True)

