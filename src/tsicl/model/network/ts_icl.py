from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from ..encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from ..icl_learning import ICLearning, ICLearningCrossAttn


class TSICLNetwork(nn.Module):
    """ TS-ICL Network: Perceiver-based encoder + Transformer ICL decoder."""

    def __init__(
        self,
        encoder: PerceiverEncoder | UnivariatePerceiverEncoder,
        head: ICLearning | ICLearningCrossAttn,
        apply_asinh_transform: bool = False,
        *args,
        **kwargs
    ):        
        super().__init__()

        self.encoder = encoder
        assert isinstance(self.encoder, PerceiverEncoder) or isinstance(self.encoder, UnivariatePerceiverEncoder)

        self.tf_icl = head
        assert isinstance(self.tf_icl, ICLearning) or isinstance(self.tf_icl, ICLearningCrossAttn)

        self.to_tf_icl = nn.Linear(self.encoder.out_dim, self.tf_icl.d_model)
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
        target_coords: torch.Tensor,
        series_covar: Optional[torch.Tensor] = None,
        coords_covar: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        sample_posterior: bool = False,
        undo_asinh_transform: bool = True,
        return_latent_only: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        

        # prepare inputs (asinh transform, etc.):
        series, series_covar = self._prepare_inputs(series, series_covar)

        # get train len for TF_ICL:
        train_size = series.shape[1]

        # encode:
        out = self.encoder(
                series             = series,
                coords             = coords,
                series_covar       = series_covar if series_covar is not None else None,
                coords_covar       = coords_covar if coords_covar is not None else None,
                mask               = mask,
                target_coords      = target_coords,
                return_stats       = return_stats,
                sample_posterior   = sample_posterior,
                return_latent_only = return_latent_only
            )
        
        if return_latent_only:
            latents = out
            assert isinstance(latents, torch.Tensor)
            return latents # (bs, num_tokens, hidden_dim)
        
        elif return_stats:
            localized_latents, kl_loss, mean, logvar = out
        
        else:
            localized_latents, kl_loss = out
        # localized_latents -> (bs, seq_len, num_bandwidth, pos_embed_dim)

        # stack freq scales:
        localized_latents = rearrange(localized_latents, 'b t s d -> b t (s d)')

        # map to ICL d_model:
        localized_latents = self.to_tf_icl(localized_latents) # (bs, seq_len, d_model)

        # ICL Transformer head:
        out = self.tf_icl(
            R       = localized_latents, # (bs, seq_len, d_model)
            y_train = series,            # (bs, context_len, 1)
            y_cov   = series_covar
        )                                # --> (bs, seq_len - context_len, 1)

        if undo_asinh_transform:
            out = self._prepare_outputs(out) # (bs, seq_len - context_len, 1)

        if return_stats:
            return out, kl_loss, mean, logvar

        return out, kl_loss