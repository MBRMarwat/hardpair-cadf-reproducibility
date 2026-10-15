"""CADF-Net v3 research implementation."""

from .config import load_config
from .model import CADFNet, FeasibilityPredictor, MLPGate
from .fusion import SoftFeasibilityFusion, build_gate_inputs

__all__ = [
    "CADFNet",
    "FeasibilityPredictor",
    "MLPGate",
    "SoftFeasibilityFusion",
    "build_gate_inputs",
    "load_config",
]
