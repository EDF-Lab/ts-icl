from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class LocalityAwareINRDecoder(nn.Module):
    """ INR decoder with quantile head taking already mutli-scale pos-embedded coords as inputs."""

    def __init__(
        self,
        output_dim: int = 1,
        embed_dim: int = 16,
        scales: List[int] = [3, 4, 5],
        dim: int = 128,
        depth: int = 3,
        start_quantile: float = 0.05,
        end_quantile: float = 0.95,
        nb_quantiles: int = 19,
        quantile_median: int = 9
    ) -> None:
        """
        Args:
            output_dim (int): output dim of the INR (*not used, replaced by quantile head*)
            embed_dim (int): size of one-scale pos embedding
            scales (List[int]): list of band freqs used for FF embeddings
            dim (int): width of the INR
            depth (int): depth of the INR
            start_quantile (float): mininmum quantile level
            end_quantile (float): maximum quantile level
            nb_quantiles (int): number of quantiles between `start_quantile` and `end_quantile`
            quantile_median (int): index of the 0.5 quantile
        """

        super().__init__()

        self.dim   = dim
        self.depth = depth

        num_scales = len(scales)

        # input proj layer:
        layers = [nn.Linear(embed_dim * num_scales, dim), nn.ReLU()]

        # add intermediate layers based on depth:
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        
        self.start_quantile  = start_quantile
        self.end_quantile    = end_quantile
        self.nb_quantiles    = nb_quantiles
        self.quantile_median = quantile_median

        # output layer:
        layers.append(nn.Linear(dim, nb_quantiles))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        localized_latents: torch.Tensor,
        return_act: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Input of size `(bs, seq_len, num_scales, dim_scale)`"""

        # stack the different scales:
        localized_latents = rearrange(localized_latents, "b n s c -> b n (s c)")

        if return_act:
            hidden_features = []
            for layer in self.mlp:
                localized_latents = layer(localized_latents)
                hidden_features.append(localized_latents)
            return localized_latents, hidden_features[:-1]

        else:

            # through the MLP:
            return self.mlp(localized_latents) # (bs, seq_len, output_dim)
