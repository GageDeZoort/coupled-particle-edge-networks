"""Training metric scheduling helpers."""

from __future__ import annotations


def heavy_metrics_epochs(max_epochs: int, frac: float = 0.0) -> set[int]:
    """
    Return 0-indexed epochs where expensive metrics (ROC AUC, bg rejection) run.

    ``frac <= 0`` → every epoch. Otherwise includes every ``frac`` fraction of
    training plus the final epoch (e.g. ``0.1`` ≈ every 10% of ``max_epochs``).
    """
    if max_epochs <= 0:
        return set()
    if frac <= 0.0:
        return set(range(max_epochs))
    interval = max(1, round(max_epochs * frac))
    epochs = {max_epochs - 1}
    for epoch in range(max_epochs):
        if (epoch + 1) % interval == 0:
            epochs.add(epoch)
    return epochs


def should_compute_heavy_metrics(epoch: int, max_epochs: int, frac: float = 0.0) -> bool:
    if frac <= 0.0:
        return True
    return epoch in heavy_metrics_epochs(max_epochs, frac)
