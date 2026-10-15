from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ErrorFocusedSpecialistLoss(nn.Module):
    def __init__(
        self,
        alpha_mel: float = 0.7,
        alpha_nv: float = 0.3,
        gamma: float = 2.0,
        error_weight: float = 3.0,
        uncertainty_weight: float = 2.0,
    ):
        super().__init__()
        self.alpha_mel = float(alpha_mel)
        self.alpha_nv = float(alpha_nv)
        self.gamma = float(gamma)
        self.error_weight = float(error_weight)
        self.uncertainty_weight = float(uncertainty_weight)

    def forward(
        self,
        specialist_logits: torch.Tensor,
        targets: torch.Tensor,
        generalist_probs: torch.Tensor,
    ) -> torch.Tensor:
        if specialist_logits.shape[0] == 0:
            return specialist_logits.sum() * 0.0
        probabilities = F.softmax(specialist_logits, dim=1)
        selected = probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-8)
        alpha = torch.where(
            targets == 0,
            torch.full_like(selected, self.alpha_mel),
            torch.full_like(selected, self.alpha_nv),
        )
        base_loss = -alpha * (1.0 - selected).pow(self.gamma) * selected.log()

        generalist_prediction = generalist_probs.argmax(dim=1)
        generalist_pair_prediction = torch.full_like(generalist_prediction, -1)
        generalist_pair_prediction[generalist_prediction == 4] = 0
        generalist_pair_prediction[generalist_prediction == 5] = 1
        generalist_wrong = generalist_pair_prediction != targets

        pair_mass = generalist_probs[:, 4] + generalist_probs[:, 5]
        conditional_mel = generalist_probs[:, 4] / pair_mass.clamp_min(1e-8)
        uncertain = (conditional_mel > 0.3) & (conditional_mel < 0.7)
        weights = torch.ones_like(selected)
        weights = weights * torch.where(generalist_wrong, self.error_weight, 1.0)
        weights = weights * torch.where(uncertain, self.uncertainty_weight, 1.0)
        return (base_loss * weights).mean()


def specialist_cross_entropy(
    specialist_logits: torch.Tensor,
    targets: torch.Tensor,
    generalist_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    del generalist_probs
    return F.cross_entropy(specialist_logits, targets)


def build_feasibility_targets(
    true_labels: torch.Tensor,
    generalist_predictions: torch.Tensor,
    mel_index: int = 4,
    off_pair_indices: tuple[int, ...] = (0, 1, 2, 3, 6),
) -> torch.Tensor:
    targets = torch.zeros(
        (len(true_labels), len(off_pair_indices)), dtype=torch.float32, device=true_labels.device
    )
    mel_mask = true_labels == int(mel_index)
    for column, class_index in enumerate(off_pair_indices):
        targets[:, column] = (mel_mask & (generalist_predictions == class_index)).float()
    return targets


def correction_rates(
    specialist_logits: torch.Tensor,
    generalist_logits: torch.Tensor,
    seven_class_labels: torch.Tensor,
) -> dict[str, float]:
    pair_mask = (seven_class_labels == 4) | (seven_class_labels == 5)
    if not pair_mask.any():
        return {"pair_conditional_correction_rate": 0.0, "full_error_correction_potential": 0.0}
    targets = (seven_class_labels[pair_mask] == 5).long()
    specialist_predictions = specialist_logits[pair_mask].argmax(dim=1)
    generalist_full = generalist_logits[pair_mask].argmax(dim=1)
    pair_mass_logits = generalist_logits[pair_mask][:, [4, 5]]
    generalist_conditional = pair_mass_logits.argmax(dim=1)

    conditional_errors = generalist_conditional != targets
    full_errors = generalist_full != seven_class_labels[pair_mask]
    specialist_correct = specialist_predictions == targets
    conditional_rate = (
        float((conditional_errors & specialist_correct).sum() / conditional_errors.sum())
        if conditional_errors.any()
        else 0.0
    )
    full_rate = (
        float((full_errors & specialist_correct).sum() / full_errors.sum())
        if full_errors.any()
        else 0.0
    )
    return {
        "pair_conditional_correction_rate": conditional_rate,
        "full_error_correction_potential": full_rate,
    }
