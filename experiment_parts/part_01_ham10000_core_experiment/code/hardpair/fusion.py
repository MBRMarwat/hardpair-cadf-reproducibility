from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import MEL_IDX, NV_IDX
from .model import LogisticGate


def pair_restricted_fusion(
    p: torch.Tensor,
    s: torch.Tensor,
    alpha: torch.Tensor | float,
    mel_idx: int = MEL_IDX,
    nv_idx: int = NV_IDX,
) -> torch.Tensor:
    p = p.float()
    s = s.float()
    if not torch.is_tensor(alpha):
        alpha = torch.full((len(p), 1), float(alpha), dtype=p.dtype, device=p.device)
    elif alpha.ndim == 1:
        alpha = alpha[:, None]
    pair = p[:, [mel_idx, nv_idx]]
    mass = pair.sum(dim=1, keepdim=True)
    conditional = pair / mass.clamp_min(1e-12)
    refined = mass * ((1.0 - alpha) * conditional + alpha * s)
    q = p.clone()
    q[:, mel_idx] = refined[:, 0]
    q[:, nv_idx] = refined[:, 1]
    return q


def gate_features(p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    pair = p[:, [MEL_IDX, NV_IDX]]
    mass = pair.sum(1, keepdim=True)
    r_mel = pair[:, :1] / mass.clamp_min(1e-12)
    offpair = [index for index in range(p.shape[1]) if index not in (MEL_IDX, NV_IDX)]
    strongest_other = p[:, offpair].max(1, keepdim=True).values
    return torch.cat([mass, r_mel, s[:, :1], (r_mel - s[:, :1]).abs(), strongest_other], 1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_temperature = nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())


def standardize_gate_features(
    fit_values: torch.Tensor, *others: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    mean = fit_values.mean(0, keepdim=True)
    std = fit_values.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std, [(values - mean) / std for values in (fit_values, *others)]


def train_gate(
    fit_p: torch.Tensor,
    fit_s: torch.Tensor,
    fit_y: torch.Tensor,
    val_p: torch.Tensor,
    val_s: torch.Tensor,
    val_y: torch.Tensor,
    config: dict,
    device: torch.device,
) -> tuple[LogisticGate, dict[str, torch.Tensor | list[float]]]:
    cfg = config["fusion"]
    fit_z = gate_features(fit_p, fit_s)
    val_z = gate_features(val_p, val_s)
    raw_fit_disagreement = (fit_z[:, 1] - fit_z[:, 2]).to(device)
    mean, std, standardized = standardize_gate_features(fit_z, val_z)
    fit_z, val_z = [value.to(device) for value in standardized]
    fit_p, fit_s, fit_y = fit_p.to(device), fit_s.to(device), fit_y.to(device)
    val_p, val_s, val_y = val_p.to(device), val_s.to(device), val_y.to(device)
    gate = LogisticGate(float(cfg["max_alpha"]), 0.5).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=float(cfg["gate_lr"]), weight_decay=float(cfg["gate_weight_decay"])
    )
    history: list[float] = []
    best_loss = float("inf")
    best_state = None
    unsuccessful = 0
    check_every = int(cfg["gate_validation_interval"])
    for step in range(1, int(cfg["gate_max_steps"]) + 1):
        gate.train()
        alpha = gate(fit_z)
        q = pair_restricted_fusion(fit_p, fit_s, alpha)
        pair_mask = (fit_y == MEL_IDX) | (fit_y == NV_IDX)
        off_mask = ~pair_mask
        pair_targets = (fit_y[pair_mask] == NV_IDX).long()
        pair_q = q[pair_mask][:, [MEL_IDX, NV_IDX]]
        pair_conditional = pair_q / pair_q.sum(1, keepdim=True).clamp_min(1e-12)
        pair_loss = nn.functional.nll_loss(pair_conditional.clamp_min(1e-12).log(), pair_targets)
        penalty = torch.zeros((), device=device)
        if off_mask.any():
            penalty = (alpha[off_mask, 0].square() * raw_fit_disagreement[off_mask].square()).mean()
        loss = pair_loss + float(cfg["offpair_penalty"]) * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % check_every == 0 or step == int(cfg["gate_max_steps"]):
            gate.eval()
            with torch.no_grad():
                val_q = pair_restricted_fusion(val_p, val_s, gate(val_z))
                val_loss = nn.functional.nll_loss(val_q.clamp_min(1e-12).log(), val_y).item()
            history.append(val_loss)
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
                unsuccessful = 0
            else:
                unsuccessful += 1
            if unsuccessful >= int(cfg["gate_patience_checks"]):
                break
    if best_state is not None:
        gate.load_state_dict(best_state)
    stats = {"mean": mean.cpu(), "std": std.cpu(), "validation_loss": history}
    return gate.cpu(), stats


def apply_gate(
    gate: LogisticGate,
    stats: dict[str, torch.Tensor],
    p: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = (gate_features(p, s) - stats["mean"]) / stats["std"]
    gate.eval()
    with torch.no_grad():
        alpha = gate(z)
        return pair_restricted_fusion(p, s, alpha), alpha

