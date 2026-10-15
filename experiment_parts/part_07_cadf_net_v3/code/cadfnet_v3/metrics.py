from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX = 4
NV_IDX = 5


def evaluate_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    values = probabilities.detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    prediction = values.argmax(axis=1)
    recalls: dict[int, float | None] = {}
    for target in (MEL_IDX, NV_IDX):
        mask = truth == target
        recalls[target] = float((prediction[mask] == target).mean()) if mask.any() else None
    valid_pair_recalls = [value for value in (recalls[MEL_IDX], recalls[NV_IDX]) if value is not None]
    pair_balanced = float(np.mean(valid_pair_recalls)) if valid_pair_recalls else None
    metrics: dict[str, Any] = {
        "images": int(len(truth)),
        "accuracy": float((prediction == truth).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=range(7), average="macro", zero_division=0)),
        "pair_balanced_accuracy": pair_balanced,
        "mel_sensitivity": recalls[MEL_IDX],
        "nv_sensitivity": recalls[NV_IDX],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=range(7)).tolist(),
    }
    pair_mass = values[:, MEL_IDX] + values[:, NV_IDX]
    strongest_offpair = values[:, [0, 1, 2, 3, 6]].max(axis=1)
    melanoma = truth == MEL_IDX
    unrecoverable = melanoma & (pair_mass < strongest_offpair)
    metrics["unrecoverable_melanoma_count"] = int(unrecoverable.sum())
    metrics["unrecoverable_melanoma_rate"] = float(unrecoverable.sum() / max(1, melanoma.sum()))
    if reference is not None:
        reference_prediction = reference.detach().cpu().numpy().argmax(axis=1)
        offpair_truth = ~np.isin(truth, [MEL_IDX, NV_IDX])
        became_pair = np.isin(prediction, [MEL_IDX, NV_IDX]) & ~np.isin(
            reference_prediction, [MEL_IDX, NV_IDX]
        )
        metrics["offpair_to_pair_transitions"] = int((offpair_truth & became_pair).sum())
        metrics["offpair_to_pair_transition_rate"] = float(
            (offpair_truth & became_pair).sum() / max(1, offpair_truth.sum())
        )
    return metrics


def pair_balanced_accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    prediction = probabilities.argmax(dim=1)
    recalls = []
    for target in (MEL_IDX, NV_IDX):
        mask = labels == target
        if mask.any():
            recalls.append(float((prediction[mask] == target).float().mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def paired_lesion_bootstrap(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    labels: torch.Tensor,
    lesion_ids: list[str],
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = np.asarray(lesion_ids)
    unique_groups = np.unique(groups)
    differences: list[float] = []
    for _ in range(int(resamples)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        differences.append(
            pair_balanced_accuracy(candidate[indices], labels[indices])
            - pair_balanced_accuracy(reference[indices], labels[indices])
        )
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    return {
        "mean_difference": float(np.nanmean(differences)),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "resamples": int(resamples),
    }
