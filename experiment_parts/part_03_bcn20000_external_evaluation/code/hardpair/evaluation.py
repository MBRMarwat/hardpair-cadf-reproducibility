from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .data import CLASS_NAMES, MEL_IDX, NV_IDX


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(1)
    predictions = probabilities.argmax(1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs((predictions[mask] == labels[mask]).mean() - confidence[mask].mean())
    return float(result)


def evaluate_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    reference_probabilities: torch.Tensor | None = None,
) -> dict[str, Any]:
    p = probabilities.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred = p.argmax(1)
    recalls = []
    for target in (MEL_IDX, NV_IDX):
        mask = y == target
        recalls.append(float((pred[mask] == target).mean()) if mask.any() else float("nan"))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(7), zero_division=0
    )
    one_hot = np.eye(7)[y]
    metrics: dict[str, Any] = {
        "pair_balanced_accuracy": float(np.nanmean(recalls)),
        "mel_sensitivity": recalls[0],
        "mel_to_nv": int(((y == MEL_IDX) & (pred == NV_IDX)).sum()),
        "nv_to_mel": int(((y == NV_IDX) & (pred == MEL_IDX)).sum()),
        "macro_f1": float(f1_score(y, pred, labels=range(7), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mel_pr_auc": float(average_precision_score((y == MEL_IDX).astype(int), p[:, MEL_IDX])),
        "brier_score": float(np.square(p - one_hot).sum(1).mean()),
        "ece_15": expected_calibration_error(p, y),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(7)).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }
    if reference_probabilities is not None:
        reference_pred = reference_probabilities.detach().cpu().numpy().argmax(1)
        offpair = ~np.isin(y, [MEL_IDX, NV_IDX])
        became_pair = np.isin(pred, [MEL_IDX, NV_IDX]) & ~np.isin(reference_pred, [MEL_IDX, NV_IDX])
        metrics["offpair_to_pair_transitions"] = int((offpair & became_pair).sum())
        metrics["offpair_to_pair_transition_rate"] = float((offpair & became_pair).sum() / max(1, offpair.sum()))
    pair_mass = p[:, MEL_IDX] + p[:, NV_IDX]
    strongest_other = np.delete(p, [MEL_IDX, NV_IDX], axis=1).max(1)
    mel_mask = y == MEL_IDX
    metrics["unrecoverable_melanoma_count"] = int((mel_mask & (pair_mass < strongest_other)).sum())
    metrics["unrecoverable_melanoma_rate"] = float(
        (mel_mask & (pair_mass < strongest_other)).sum() / max(1, mel_mask.sum())
    )
    return metrics


def paired_group_bootstrap(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    labels: torch.Tensor,
    lesion_ids: list[str],
    resamples: int,
    seed: int,
) -> dict[str, float]:
    def pair_balanced_accuracy(probabilities: torch.Tensor, target: torch.Tensor) -> float:
        prediction = probabilities.argmax(1)
        recalls: list[float] = []
        for class_index in (MEL_IDX, NV_IDX):
            mask = target == class_index
            if mask.any():
                recalls.append(float((prediction[mask] == class_index).float().mean()))
        return float(np.mean(recalls)) if recalls else float("nan")

    rng = np.random.default_rng(seed)
    groups = np.asarray(lesion_ids)
    unique_groups = np.unique(groups)
    differences: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        candidate_score = pair_balanced_accuracy(candidate[indices], labels[indices])
        baseline_score = pair_balanced_accuracy(baseline[indices], labels[indices])
        differences.append(candidate_score - baseline_score)
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    return {
        "mean_difference": float(np.nanmean(differences)),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
    }

