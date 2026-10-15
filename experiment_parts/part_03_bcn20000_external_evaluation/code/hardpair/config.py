from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["_config_path"] = str(path)
    config["_root"] = str(path.parent)
    for key in ("metadata_csv", "output_dir"):
        value = Path(config["paths"][key])
        config["paths"][key] = str(value if value.is_absolute() else path.parent / value)
    config["paths"]["image_dirs"] = [
        str(Path(value) if Path(value).is_absolute() else path.parent / value)
        for value in config["paths"]["image_dirs"]
    ]
    return config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda")).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)

