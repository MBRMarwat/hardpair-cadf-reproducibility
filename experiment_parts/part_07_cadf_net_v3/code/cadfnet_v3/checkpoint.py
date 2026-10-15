from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class HourlyCheckpoint:
    def __init__(self, path: str | Path, interval_seconds: float = 3600.0):
        self.path = Path(path)
        self.interval_seconds = float(interval_seconds)
        self.last_save = time.monotonic()

    def due(self) -> bool:
        return time.monotonic() - self.last_save >= self.interval_seconds

    def save(self, payload: Any) -> None:
        atomic_torch_save(payload, self.path)
        self.last_save = time.monotonic()

    def load(self, map_location: str | torch.device = "cpu") -> Any:
        return torch.load(self.path, map_location=map_location)
