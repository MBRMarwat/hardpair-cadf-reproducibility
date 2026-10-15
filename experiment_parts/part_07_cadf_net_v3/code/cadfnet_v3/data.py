from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
MEL_IDX = 4
NV_IDX = 5
OFF_PAIR_INDICES = [0, 1, 2, 3, 6]
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


class HeuristicLesionCropper:
    """Deterministic label-free cropper used because segmentation masks are unavailable."""

    def __init__(self, margin: float = 0.15):
        self.margin = float(margin)

    def __call__(self, image: Image.Image) -> Image.Image:
        original = image.convert("RGB")
        preview = original.copy()
        preview.thumbnail((128, 128), Image.Resampling.BILINEAR)
        array = np.asarray(preview, dtype=np.float32) / 255.0
        height, width, _ = array.shape
        border = np.concatenate(
            [array[0], array[-1], array[:, 0], array[:, -1]], axis=0
        )
        border_color = np.median(border, axis=0)
        color_distance = np.linalg.norm(array - border_color, axis=2)
        darkness = 1.0 - array.mean(axis=2)
        saturation = array.max(axis=2) - array.min(axis=2)
        yy, xx = np.mgrid[0:height, 0:width]
        radius = np.sqrt(
            ((xx - (width - 1) / 2) / max(width / 2, 1)) ** 2
            + ((yy - (height - 1) / 2) / max(height / 2, 1)) ** 2
        )
        center_prior = np.clip(1.15 - radius, 0.0, 1.0)
        score = (0.55 * color_distance + 0.30 * darkness + 0.15 * saturation) * (
            0.55 + 0.45 * center_prior
        )
        central_values = score[radius <= 0.95]
        threshold = float(np.quantile(central_values, 0.65)) if central_values.size else float(score.mean())
        mask = (score >= threshold) & (radius <= 1.0)
        ys, xs = np.nonzero(mask)
        if len(xs) < max(16, int(0.01 * width * height)):
            left, top, right, bottom = 0.1, 0.1, 0.9, 0.9
        else:
            left, right = xs.min() / width, (xs.max() + 1) / width
            top, bottom = ys.min() / height, (ys.max() + 1) / height
            box_width, box_height = right - left, bottom - top
            left -= self.margin * box_width
            right += self.margin * box_width
            top -= self.margin * box_height
            bottom += self.margin * box_height
            left, top = max(0.0, left), max(0.0, top)
            right, bottom = min(1.0, right), min(1.0, bottom)
        ow, oh = original.size
        return original.crop(
            (
                int(round(left * ow)),
                int(round(top * oh)),
                max(1, int(round(right * ow))),
                max(1, int(round(bottom * oh))),
            )
        )


class ColorDecomposition:
    def __init__(self):
        self.matrix = torch.tensor(
            [[0.85, 0.15, 0.00], [0.10, 0.80, 0.10]], dtype=torch.float32
        )

    @staticmethod
    def _normalize(channel: torch.Tensor) -> torch.Tensor:
        minimum = channel.amin(dim=(-2, -1), keepdim=True)
        maximum = channel.amax(dim=(-2, -1), keepdim=True)
        return (channel - minimum) / (maximum - minimum + 1e-8)

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W] RGB tensor; got {tuple(image.shape)}")
        optical_density = -torch.log(image.clamp(1e-6, 1.0)).reshape(3, -1)
        decomposed = self.matrix.to(image.device) @ optical_density
        height, width = image.shape[-2:]
        melanin = self._normalize(decomposed[0].reshape(1, height, width))
        hemoglobin = self._normalize(decomposed[1].reshape(1, height, width))
        return melanin, hemoglobin


def color_statistics(
    rgb: torch.Tensor, melanin: torch.Tensor, hemoglobin: torch.Tensor
) -> torch.Tensor:
    channels = torch.cat([rgb, melanin, hemoglobin], dim=0)
    means = channels.mean(dim=(-2, -1))
    standard_deviations = channels.std(dim=(-2, -1), unbiased=False)
    melanin_q90 = torch.quantile(melanin.flatten(), 0.90)
    hemoglobin_q90 = torch.quantile(hemoglobin.flatten(), 0.90)
    values = torch.cat([means, standard_deviations, melanin_q90[None], hemoglobin_q90[None]])
    if values.shape != (12,):
        raise AssertionError(f"Color statistics must have 12 values; got {tuple(values.shape)}")
    return values


class CADFDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int = 192,
        training: bool = False,
        same_view: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.training = bool(training)
        self.same_view = bool(same_view)
        self.resize = ResizePad(image_size)
        self.cropper = HeuristicLesionCropper()
        self.decompose = ColorDecomposition()
        self.to_tensor = transforms.ToTensor()
        self.augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15, fill=(128, 128, 128)),
                transforms.ColorJitter(brightness=0.05, contrast=0.10),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as opened:
            image = opened.convert("RGB")
            full = self.resize(image)
            lesion = self.resize(self.cropper(image))
        if self.training:
            full = self.augment(full)
            lesion = self.augment(lesion)
        full_rgb = self.to_tensor(full)
        lesion_rgb = self.to_tensor(lesion)
        melanin, hemoglobin = self.decompose(lesion_rgb)
        stats = color_statistics(lesion_rgb, melanin, hemoglobin)

        full_normalized = (full_rgb - RGB_MEAN) / RGB_STD
        generalist_view = torch.cat(
            [full_normalized, torch.zeros(2, *full_normalized.shape[-2:])], dim=0
        )
        if self.same_view:
            specialist_view = generalist_view.clone()
        else:
            lesion_normalized = (lesion_rgb - RGB_MEAN) / RGB_STD
            decomposition = (torch.cat([melanin, hemoglobin], dim=0) - 0.5) / 0.25
            specialist_view = torch.cat([lesion_normalized, decomposition], dim=0)

        return {
            "generalist_view": generalist_view,
            "specialist_view": specialist_view,
            "color_stats": stats,
            "label": int(row.label),
            "image_id": str(row.image_id),
            "lesion_id": str(row.lesion_id),
        }


class ResumableRandomSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sequence[Any],
        seed: int,
        epoch: int,
        start_index: int = 0,
    ):
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        indices = torch.randperm(len(data_source), generator=generator).tolist()
        self.indices = indices[int(start_index) :]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    training: bool,
    batch_size: int,
    epoch: int = 0,
    start_index: int = 0,
    same_view: bool = False,
) -> DataLoader:
    dataset = CADFDataset(
        frame,
        image_size=int(config["data"]["image_size"]),
        training=training,
        same_view=same_view,
    )
    sampler = (
        ResumableRandomSampler(dataset, config["seed"], epoch, start_index)
        if training
        else None
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"].get("pin_memory", True) and torch.cuda.is_available()),
        worker_init_fn=seed_worker,
        persistent_workers=int(config["data"]["num_workers"]) > 0,
    )


def load_ham_manifest(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(config["paths"]["split_manifest"])
    required = {"image_id", "lesion_id", "label", "split", "image_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if not frame["image_path"].map(lambda value: Path(value).is_file()).all():
        lookup: dict[str, str] = {}
        for directory in config["paths"]["ham_image_dirs"]:
            for path in Path(directory).glob("*.jpg"):
                lookup[path.stem] = str(path.resolve())
        frame["image_path"] = frame["image_id"].map(lookup)
    if frame["image_path"].isna().any():
        raise FileNotFoundError("One or more HAM10000 images could not be resolved")
    leakage = frame.groupby("lesion_id")["split"].nunique().max()
    if int(leakage) != 1:
        raise AssertionError("Lesion leakage detected across HAM10000 splits")
    return frame


def stratified_smoke_sample(frame: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_images:
        return frame.copy().reset_index(drop=True)
    groups = frame.groupby(["split", "label"], sort=True).groups
    if max_images < len(groups):
        raise ValueError("Smoke sample is too small to include each split/class stratum")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    remaining: list[int] = []
    for indices in groups.values():
        values = np.asarray(list(indices), dtype=int)
        rng.shuffle(values)
        selected.append(int(values[0]))
        remaining.extend(values[1:].tolist())
    rng.shuffle(remaining)
    selected.extend(remaining[: max_images - len(selected)])
    return frame.loc[selected].sort_values(["split", "label", "image_id"]).reset_index(drop=True)


def load_bcn_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(config["paths"]["bcn_metadata"])
    label_map = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}
    metadata["dx"] = metadata["diagnosis"].map(label_map)
    metadata["image_id"] = metadata["bcn_filename"].map(lambda value: Path(value).stem)
    metadata["image_path"] = metadata["bcn_filename"].map(
        lambda value: str((Path(config["paths"]["bcn_image_dir"]) / value).resolve())
    )
    primary = metadata.loc[metadata["dx"].notna()].copy()
    primary["label"] = primary["dx"].map(CLASS_TO_IDX).astype(int)
    excluded = metadata.loc[metadata["dx"].isna()].copy()
    return primary.reset_index(drop=True), excluded.reset_index(drop=True)


def effective_number_class_weights(labels: torch.Tensor, classes: int, beta: float) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=classes).float()
    effective = 1.0 - torch.pow(torch.tensor(float(beta)), counts)
    weights = (1.0 - float(beta)) / effective.clamp_min(1e-12)
    weights[counts == 0] = 0.0
    positive = weights > 0
    weights[positive] = weights[positive] / weights[positive].mean()
    return weights
