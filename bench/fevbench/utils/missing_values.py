"""Model-agnostic masking utilities for missing-value robustness experiments.

These helpers are intentionally pure NumPy and free of any model/framework
imports so that the same gap-generation logic can be reused across FEVBench
runners (Chronos-2, TabPFN, TabICL, naive baselines, ...).

Two public functions are exposed:

* ``derive_series_rng`` produces a reproducible ``np.random.Generator`` for a
  given ``(task_idx, window_idx, series_idx)`` triple, given a global seed.
  Two runs that share the same ``base_seed`` and benchmark task ordering will
  produce bit-identical RNG states for every series, regardless of the
  masking ratio used.

* ``apply_pointwise_mask`` masks a fixed fraction of positions of a 1-D
  target array uniformly at random (without replacement) by replacing the
  selected entries with ``np.nan``.
"""

from __future__ import annotations

import numpy as np


def derive_series_rng(
    base_seed: int,
    task_idx: int,
    window_idx: int,
    series_idx: int,
) -> np.random.Generator:
    """Return a reproducible Generator for one ``(task, window, series)`` triple.

    Uses ``np.random.SeedSequence`` with a ``spawn_key`` tuple so the triple
    directly contributes entropy to the child seed. The derivation is
    collision-free for any ``(task_idx, window_idx, series_idx)`` triple and
    bit-identical across re-runs with the same ``base_seed``.

    Parameters
    ----------
    base_seed:
        The global benchmark seed (typically ``cfg.seed``).
    task_idx:
        0-based position of the task within the (primed) benchmark task list.
    window_idx:
        0-based position of the rolling window within the task.
    series_idx:
        0-based position of the series within the window's ``past_data``
        dataset.
    """
    ss = np.random.SeedSequence(
        int(base_seed),
        spawn_key=(int(task_idx), int(window_idx), int(series_idx)),
    )
    return np.random.default_rng(ss)


def apply_pointwise_mask(
    target: np.ndarray,
    missing_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of ``target`` with a fraction of positions set to ``NaN``.

    Exactly ``round(missing_ratio * T)`` positions are sampled uniformly
    without replacement via ``rng.choice`` and replaced with ``np.nan``.
    A ratio of ``0.0`` is a clean no-op (returns a copy unchanged).

    Parameters
    ----------
    target:
        1-D float array of shape ``(T,)``. Pre-existing NaNs are preserved;
        masking is applied to all positions to keep the effective ratio
        exact.
    missing_ratio:
        Fraction of positions to mask, in ``[0, 1]``.
    rng:
        A ``numpy.random.Generator`` (e.g. from ``derive_series_rng``).
    """
    if not (0.0 <= float(missing_ratio) <= 1.0):
        raise ValueError(
            f"missing_ratio must be in [0, 1], got {missing_ratio}"
        )
    if target.ndim != 1:
        raise ValueError(
            f"target must be 1-D, got shape {target.shape}"
        )

    masked = target.astype(np.float32, copy=True)
    seq_len = masked.shape[0]
    if seq_len == 0:
        return masked

    n_mask = int(round(float(missing_ratio) * seq_len))
    n_mask = min(seq_len, max(0, n_mask))
    if n_mask > 0:
        chosen = rng.choice(seq_len, size=n_mask, replace=False)
        masked[chosen] = np.nan
    return masked
