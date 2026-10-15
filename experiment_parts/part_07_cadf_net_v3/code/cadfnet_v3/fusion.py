from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


MEL_IDX = 4
NV_IDX = 5
OFF_PAIR_INDICES = (0, 1, 2, 3, 6)


def _normalized_binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    pair = torch.stack([probability, 1.0 - probability], dim=1).clamp_min(1e-8)
    return -(pair * pair.log()).sum(dim=1) / math.log(2.0)


def build_gate_inputs(
    probabilities: torch.Tensor,
    specialist_conditional: torch.Tensor,
    feasibility: torch.Tensor,
    off_pair_indices: tuple[int, ...] = OFF_PAIR_INDICES,
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] != 7:
        raise ValueError("Generalist probabilities must have shape [B,7]")
    if specialist_conditional.shape != (probabilities.shape[0], 2):
        raise ValueError("Specialist probabilities must have shape [B,2]")
    if feasibility.shape != (probabilities.shape[0], len(off_pair_indices)):
        raise ValueError("Feasibility scores must have shape [B,5]")
    pair_mass = probabilities[:, MEL_IDX] + probabilities[:, NV_IDX]
    conditional_mel = probabilities[:, MEL_IDX] / pair_mass.clamp_min(1e-8)
    specialist_mel = specialist_conditional[:, 0]
    generalist_entropy = _normalized_binary_entropy(conditional_mel)
    specialist_entropy = _normalized_binary_entropy(specialist_mel)
    disagreement = (conditional_mel - specialist_mel).abs()
    off_pair = probabilities[:, list(off_pair_indices)]
    strongest_probability, strongest_column = off_pair.max(dim=1)
    strongest_feasibility = feasibility.gather(1, strongest_column[:, None]).squeeze(1)
    return torch.stack(
        [
            pair_mass.clamp(0.0, 1.0),
            conditional_mel.clamp(0.0, 1.0),
            specialist_mel.clamp(0.0, 1.0),
            generalist_entropy.clamp(0.0, 1.0),
            specialist_entropy.clamp(0.0, 1.0),
            disagreement.clamp(0.0, 1.0),
            strongest_probability.clamp(0.0, 1.0),
            strongest_feasibility.clamp(0.0, 1.0),
        ],
        dim=1,
    )


class SoftFeasibilityFusion(nn.Module):
    def __init__(self, kappa: float = 0.3, temperature: float = 0.1):
        super().__init__()
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("kappa must be in [0,1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.kappa = float(kappa)
        self.temperature = float(temperature)

    def forward(
        self,
        probabilities: torch.Tensor,
        specialist_conditional: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        feasibility: torch.Tensor,
    ) -> torch.Tensor:
        batch = probabilities.shape[0]
        if alpha.shape != (batch,) or beta.shape != (batch,):
            raise ValueError("alpha and beta must each have shape [B]")
        pair_mass = probabilities[:, MEL_IDX] + probabilities[:, NV_IDX]
        conditional_mel = probabilities[:, MEL_IDX] / pair_mass.clamp_min(1e-8)
        conditional_nv = probabilities[:, NV_IDX] / pair_mass.clamp_min(1e-8)
        primary_mel = pair_mass * (
            (1.0 - alpha) * conditional_mel + alpha * specialist_conditional[:, 0]
        )
        primary_nv = pair_mass * (
            (1.0 - alpha) * conditional_nv + alpha * specialist_conditional[:, 1]
        )

        off_pair = probabilities[:, list(OFF_PAIR_INDICES)]
        strongest, strongest_column = off_pair.max(dim=1)
        second = off_pair.topk(k=2, dim=1).values[:, 1]
        strongest_feasibility = feasibility.gather(1, strongest_column[:, None]).squeeze(1)
        feasibility_gate = torch.sigmoid((strongest_feasibility - 0.5) / self.temperature)
        alpha_gate = torch.sigmoid((alpha - 0.5) / self.temperature)
        melanoma_preference = torch.sigmoid((primary_mel - primary_nv) / self.temperature)
        maximum_transfer = torch.minimum(strongest - second, self.kappa * strongest).clamp_min(0.0)
        delta = beta * maximum_transfer * feasibility_gate * alpha_gate * melanoma_preference

        primary_adjustment = torch.zeros_like(probabilities)
        primary_adjustment[:, MEL_IDX] = primary_mel - probabilities[:, MEL_IDX]
        primary_adjustment[:, NV_IDX] = primary_nv - probabilities[:, NV_IDX]
        strongest_full_index = torch.tensor(
            OFF_PAIR_INDICES, device=probabilities.device, dtype=torch.long
        )[strongest_column]
        transfer_adjustment = F.one_hot(
            torch.full_like(strongest_full_index, MEL_IDX), num_classes=7
        ).to(probabilities.dtype) * delta[:, None]
        transfer_adjustment = transfer_adjustment - F.one_hot(
            strongest_full_index, num_classes=7
        ).to(probabilities.dtype) * delta[:, None]
        fused = probabilities + primary_adjustment + transfer_adjustment
        fused = fused.clamp_min(1e-8)
        return fused / fused.sum(dim=1, keepdim=True)
