from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def cache_fn(f: Callable) -> Callable:
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def dropout_seq(
    images: torch.Tensor,
    coordinates: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.25
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Randomly drop some elements of tensors along the sequence dimension.

    Args:
        images (torch.Tensor): values tensor of shape `(bs, seq_len, ...)`
        coordinates (torch.Tensor): tensor of coordinates of shape `(bs, seq_len, ...)`
        mask (torch.Tensor): boolean mask of shape `(bs, seq_len, ...)`
        dropout (float): dropout fraction
    
    Returns:
        Same input tensors with a `dropout` fraction of elements removed from the sequence dim
    """
    
    b, n, *_, device = *images.shape, images.device

    # get the indices of the elements to dropout (along seq dim):
    logits       = torch.randn(b, n, device=device)
    keep_prob    = 1.0 - dropout
    num_keep     = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices  # (bs, num_keep)

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1") # (bs, 1)

    if mask is None:
        images      = images[batch_indices, keep_indices]       # (bs, num_keep, ...)
        coordinates = coordinates[batch_indices, keep_indices]  # (bs, num_keep, ...)

        return images, coordinates

    else:
        images      = images[batch_indices, keep_indices]
        coordinates = coordinates[batch_indices, keep_indices]
        mask        = mask[batch_indices, keep_indices]

        return images, coordinates, mask


class DiagonalGaussianDistribution(object):

    def __init__(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False
    ) -> None:

        self.mean = mean
        self.deterministic = deterministic
        
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )
        else:
            self.logvar = logvar
            self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    def sample(self, K: int = 1) -> torch.Tensor:
        """ Draw `K`samples from the Diag Gaussian distrib."""

        if K == 1:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(
                device=self.mean.device
            )
            return x
        else:
            # XXX check dims in repeat for time series
            x = self.mean[None, ...].repeat([K, 1, 1, 1]) + self.std[None, ...].repeat(
                K, 1, 1, 1
            ) * torch.randn([K, *self.mean.shape]).to(device=self.mean.device)
            return x

    def kl(self, other = None) -> torch.Tensor:
        """ Compute KL divergence from object to another instance of `DiagonalGaussianDistribution`"""

        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2]
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2],
                )

    def nll(
        self,
        sample: torch.Tensor,
        dims: Sequence = [1, 2]
    ) -> torch.Tensor:
        
        if self.deterministic:
            return torch.Tensor([0.0])
        
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


def linear_scheduler(
    start: float,
    end: float,
    num_steps: int
) -> Sequence[float]:
    delta = (end - start) / num_steps
    return [start + i * delta for i in range(num_steps)]

