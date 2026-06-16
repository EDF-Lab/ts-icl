import abc
from typing import Optional, Union

import torch


class AbstractCustomScaler(abc.ABC):

    @abc.abstractmethod
    def fit(
        self,
        X: torch.Tensor
    ) -> None: ...

    @abc.abstractmethod
    def transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor: ...

    @abc.abstractmethod
    def inv_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor: ...


class CustomStandardScaler(AbstractCustomScaler):

    def __init__(
        self,
        dim: Union[int, tuple] = 1,
        epsilon: float = 1e-5
    ) -> None:

        self.mean    = torch.empty((1,))
        self.std     = torch.empty((1,))
        self.dim     = dim
        self.epsilon = epsilon
        
    def fit(
        self,
        X: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> None:
         
        # X has shape (batch_size, grid_size, d) (in general: d=1)
        
        if mask is not None:
            assert mask.shape == X.shape
            x = X.clone()
            x[mask] = torch.nan
        else:
            x = X

        self.mean: torch.Tensor = torch.nan_to_num( torch.nanmean(x, dim=self.dim, keepdim=True), nan=0.0)
        scale = torch.nan_to_num( (x - self.mean).square().nanmean(dim=self.dim, keepdim=True).sqrt(), nan=1.0)
        self.std: torch.Tensor  = torch.where(scale==0, self.epsilon, scale)
        # self.std: torch.Tensor  = torch.std(X, dim=self.dim, keepdim=True) + self.epsilon

    def transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_transform = (X - self.mean) / self.std

        return X_transform
    
    def inv_transform(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if mask is not None:
            std  = self.std[mask]
            mean = self.mean[mask]
        else:
            std  = self.std
            mean = self.mean

        X_inv_transform = X * std + mean

        return X_inv_transform


class CustomMinmaxScaler(AbstractCustomScaler):

    def __init__(
        self,
        dim: int = 1,
        epsilon: float = 1e-8
    ) -> None:

        self.xmin    = torch.empty((1,))
        self.xmax    = torch.empty((1,))
        self.dim     = dim
        self.epsilon = epsilon
        
    def fit(
        self,
        X: torch.Tensor
    ) -> None:
         
        # X has shape (batch_size, grid_size, d) (in general: d=1)
        self.xmin: torch.Tensor = torch.min(X, dim=self.dim, keepdim=True)[0]
        self.xmax: torch.Tensor = torch.max(X, dim=self.dim, keepdim=True)[0]

    def transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_transform = (X - self.xmin) / (self.xmax - self.xmin + self.epsilon)

        return X_transform
    
    def inv_transform(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:

        X_inv_transform = (self.xmax - self.xmin + self.epsilon) * X + self.xmin

        return X_inv_transform