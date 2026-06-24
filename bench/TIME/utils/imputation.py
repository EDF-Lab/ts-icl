import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality


def _infer_season_length(freq: str) -> int:
    try:
        return int(get_seasonality(freq))
    except Exception:
        return 1


def get_max_context_length(term: str) -> int:

    if term == 'short':
        return 256
    elif term == 'medium':
        return 1024
    elif term == 'long':
        return 4096
    else:
        raise NotImplementedError


def infer_steps_per_day(freq: str | None) -> int:
    """
    Infer how many time steps correspond to one day from dataset.freq.
    Falls back to 1 if the frequency is not a fixed sub-daily/daily offset.
    """
    if freq is None:
        return 1

    try:
        offset = pd.tseries.frequencies.to_offset(str(freq))
        nanos = offset.nanos
        day_nanos = pd.Timedelta(days=1).value
        if nanos > 0 and day_nanos % nanos == 0:
            return max(1, int(day_nanos // nanos))
    except Exception:
        pass

    return 1


def build_default_imputation_scenarios(context_length: int) -> list[dict]:

    # We define the block length as roughly 1/15 of the available context.
    # This creates contiguous missing segments that are large enough to be
    # meaningful, while remaining proportional to the sequence length.

    block_size = max(1, min(context_length, context_length // 15))

    return [
        {
            "name": "scenario_blocks_missing_1",
            # Single-block missingness:
            # one contiguous gap is removed from the context.
            "missing_pointwise_ratio": 0.0,
            "num_blocks": 1,
            "block_size": block_size,
        },
        {
            "name": "scenario_blocks_missing_2",
            # Double-block missingness:
            # two separate contiguous gaps are removed from the context.
            "missing_pointwise_ratio": 0.0,
            "num_blocks": 2,
            "block_size": block_size,
        },
        {
            "name": "scenario_pointwise_missing_1",
            # Moderate pointwise missingness:
            # 50% of timestamps are removed independently.
            "missing_pointwise_ratio": 0.50,
            "num_blocks": 0,
            "block_size": 0,
        },
        {
            "name": "scenario_pointwise_missing_2",
            # Severe pointwise missingness:
            # 70% of timestamps are removed independently.
            "missing_pointwise_ratio": 0.70,
            "num_blocks": 0,
            "block_size": 0,
        },
    ]


def sample_time_mask(
    seq_len: int,
    rng: np.random.Generator,
    missing_pointwise_ratio: float = 0.0,
    num_blocks: int = 0,
    block_size: int = 0,
) -> np.ndarray:
    
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    mask = np.zeros(seq_len, dtype=bool)

    # 1) Block missing
    if num_blocks > 0:

        block_size = max(1, min(int(block_size), seq_len))

        for _ in range(num_blocks):
            placed = False
            for _attempt in range(100):
                start = int(rng.integers(0, seq_len - block_size + 1))
                if not mask[start:start + block_size].any():
                    mask[start:start + block_size] = True
                    placed = True
                    break

            if not placed:
                start = int(rng.integers(0, seq_len - block_size + 1))
                mask[start:start + block_size] = True
            
    # 2) Pointwise missing
    if missing_pointwise_ratio > 0:
        if not (0.0 <= missing_pointwise_ratio <= 1.0):
            raise ValueError(
                f"missing_pointwise_ratio must be in [0, 1], got {missing_pointwise_ratio}"
            )

        num_points = int(round(missing_pointwise_ratio * seq_len))
        num_points = min(seq_len, max(1, num_points))

        available = np.flatnonzero(~mask)
        num_points = min(num_points, available.size)

        if num_points > 0:
            chosen = rng.choice(available, size=num_points, replace=False)
            mask[chosen] = True

    # Fallback only if the scenario asked for nothing
    if not mask.any():
        mask[int(rng.integers(0, seq_len))] = True

    return mask



def prepare_context(
    d,
    rng: np.random.Generator,
    context_length: int,
    scenario: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one masked context according to the given scenario.

    Returns
    -------
    masked_target : torch.Tensor
        Shape (t, c), with NaNs at missing positions.
    target_t : torch.Tensor
        Shape (t, c), original values.
    missing_mask : torch.Tensor
        Shape (t, c), True where values were masked.
    """
    target = np.asarray(d["target"], dtype=np.float32)

    if target.ndim == 1:
        target = target[np.newaxis, :]

    if target.shape[-1] > context_length:
        target = target[..., -context_length:]

    # Convert from (c, t) -> (t, c)
    target_t = target.T.astype(np.float32)

    time_mask = sample_time_mask(
        seq_len=target_t.shape[0],
        rng=rng,
        missing_pointwise_ratio=float(scenario.get("missing_pointwise_ratio", 0.0)),
        num_blocks=int(scenario.get("num_blocks", 0)),
        block_size=int(scenario.get("block_size", 0)),
    )

    # Same timestamps masked for all channels
    missing_mask = np.broadcast_to(time_mask[:, None], target_t.shape).copy()

    masked_target = target_t.copy()
    masked_target[missing_mask] = np.nan

    return (
        torch.tensor(masked_target, dtype=torch.float32),
        torch.tensor(target_t, dtype=torch.float32),
        torch.tensor(missing_mask, dtype=torch.bool),
    )
