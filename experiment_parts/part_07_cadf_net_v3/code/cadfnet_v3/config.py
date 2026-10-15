from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str | Path,
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if override_path is not None:
        with Path(override_path).resolve().open("r", encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle))

    part_dir = config_path.parent.parent
    config["part_dir"] = str(part_dir)
    for key in (
        "split_manifest",
        "ham_metadata",
        "bcn_metadata",
        "bcn_image_dir",
        "output_dir",
    ):
        config["paths"][key] = str((part_dir / config["paths"][key]).resolve())
    config["paths"]["ham_image_dirs"] = [
        str((part_dir / value).resolve()) for value in config["paths"]["ham_image_dirs"]
    ]
    return config
