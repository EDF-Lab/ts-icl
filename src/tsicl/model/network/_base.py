from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from ..inr import LocalityAwareINRDecoder


class PerceiverINR(nn.Module):
    """ Perceiver TS encoder + INR decoder."""
    
    def __init__(
        self,
        encoder: PerceiverEncoder | UnivariatePerceiverEncoder,
        head: LocalityAwareINRDecoder,
        apply_asinh_transform: bool = True,
        *args,
        **kwargs
    ):
        
        super().__init__()

        self.encoder = encoder
        assert isinstance(self.encoder, PerceiverEncoder) or isinstance(self.encoder, UnivariatePerceiverEncoder)

        self.decoder = head
        assert isinstance(self.decoder, LocalityAwareINRDecoder)

        self.apply_asinh_transform = apply_asinh_transform

    def _prepare_inputs(
        self,
        series: torch.Tensor,
        series_covar: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        
        if self.apply_asinh_transform:
            series = torch.asinh(series)
            if series_covar is not None:
                series_covar = torch.asinh(series_covar)
        
        return series, series_covar
    
    def _prepare_outputs(
        self,
        series: torch.Tensor
    ) -> torch.Tensor:
        
        if self.apply_asinh_transform:
            series = torch.sinh(series)
        
        return series

    def forward(
        self,
        series: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_coords: Optional[torch.Tensor] = None,
        series_covar: Optional[torch.Tensor] = None,
        coords_covar: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        sample_posterior: bool = True,
        return_act: bool = False,
        undo_asinh_transform: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # prepare inputs (asinh transform, etc.):
        series, series_covar = self._prepare_inputs(series, series_covar)

        # encode:
        out = self.encoder(
            series           = series,
            coords           = coords,
            series_covar     = series_covar if series_covar is not None else None,
            coords_covar     = coords_covar if coords_covar is not None else None,    
            mask             = mask,
            target_coords    = target_coords,
            return_stats     = return_stats,
            sample_posterior = sample_posterior,
        )
        
        if return_stats:
            localized_latents, kl_loss, mean, logvar = out
        else:
            localized_latents, kl_loss = out
        # localized_latents -> (bs, target_seq_len, num_bandwidth, pos_embed_dim)

        # 
        output_features = self.decoder(localized_latents, return_act = return_act) # (bs, target_seq_len, num_channels)

        if undo_asinh_transform:
            output_features = self._prepare_outputs(output_features) # (bs, seq_len - context_len, 1)
        
        if return_act:
            return output_features

        if return_stats:
            return output_features, kl_loss, mean, logvar

        return output_features, kl_loss
