import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., in_features)  
        return: torch.Tensor of shape (..., out_features)
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: torch.Tensor of shape (...), with integer values in [0, num_embeddings)
        return: torch.Tensor of shape (..., embedding_dim)
        """
        return self.weight[token_ids] # Replace each token id with its embedding vector