import torch
import torch.nn as nn

from .gaussian import GaussianEncoding
from .nerf import NeRFEncoding


class FourierPositionalEmbedding(nn.Module):
    """ Fourier Feature layer."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_freq: int = 32,
        max_freq_log2: int = 5,
        input_dim: int = 2,
        base_freq: int = 2,
        use_relu: bool = True,
    ) -> None:
        
        super().__init__()

        self.nerf_embedder = NeRFEncoding(
            num_freq      = num_freq,
            max_freq_log2 = max_freq_log2,
            min_freq_log2 = 0,
            input_dim     = input_dim,
            base_freq     = base_freq,
            log_sampling  = False,
            include_input = True
        )

        self.linear = nn.Linear(self.nerf_embedder.out_dim, hidden_dim)
        self.use_relu = use_relu

    def forward(self, coords: torch.Tensor) -> torch.Tensor:

        x = self.nerf_embedder(coords)
        if self.use_relu:
            x = torch.relu(self.linear(x))
        else:
            x = self.linear(x)  # try without relu

        return x

class FourierFeatures(nn.Module):
    
    """
    INR for a single instance
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        num_frequencies = 8,
        width: int = 256,
        depth: int = 5,
        frequency_embedding: str = "nerf",
        include_input: bool = True,
        scale: int = 5,
        log_sampling: bool = False,
        min_frequencies: int = 0,
        max_frequencies: int = 32,
        base_frequency: float | int = 1.25,
    ) -> None:
        
        super().__init__()

        self.frequency_embedding = frequency_embedding.lower() # type of frequency embedding (NeRF or Gaussian)
        self.include_input = include_input                     # whether to add coordinate (time) to freq embedding

        # (time) coordinate embedding with NeRF:
        if self.frequency_embedding == "nerf":
            self.embedding = NeRFEncoding(
                num_freq      = num_frequencies,
                min_freq_log2 = min_frequencies,
                max_freq_log2 = max_frequencies,
                log_sampling  = log_sampling,
                include_input = include_input,
                input_dim     = input_dim,
                base_freq     = base_frequency
            )
            self.in_channels = [self.embedding.out_dim] + [width] * (depth - 1)

        # or (time) coordinate embedding with Gaussian encoding:
        elif self.frequency_embedding == "gaussian":
            self.scale = scale
            self.embedding = GaussianEncoding(
                embedding_size=num_frequencies * 2, scale=scale, dims=input_dim
            )
            embed_dim = (
                num_frequencies * 2 + input_dim
                if include_input
                else num_frequencies * 2
            )
            self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [output_dim]
        
        # INR layers (linear):
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )

        # corresponding INR activations:
        self.activations = nn.ModuleList(
            [nn.ReLU() for k in range(depth-1)]
        )

        self.depth      = depth # depth of the network
        self.hidden_dim = width # hidden dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], dim=-1)

        for idx, l in enumerate(self.layers[:-1]):
            position = self.activations[idx](l(position))

        out = self.layers[-1](position)

        return out

if __name__ == "__main__":

    model = FourierFeatures()
    X = torch.rand(1, 256, 1)
    out = model(X)

