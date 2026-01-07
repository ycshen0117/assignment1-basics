import torch
import torch.nn as nn
from einops import rearrange, einsum, repeat


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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}

        self.scale = nn.Parameter(torch.ones((d_model,), **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., d_model)
        return: torch.Tensor of shape (..., d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized_x = x * torch.rsqrt(mean_square + self.eps)
        result = normalized_x * self.scale
        return result.to(in_dtype)
    

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs)
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., d_model)
        return: torch.Tensor of shape (..., d_model)
        """
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for Rotary Positional Embedding.")
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos_cached = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
        sin_cached = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.max_seq_len = max_seq_len
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
        return x * cos + x_rot * sin
        

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") * (d_k ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)
    return einsum(attn, V, "... q k, ... k d -> ... q d")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: torch.Tensor of shape (..., sequence_length, d_model)
        return: torch.Tensor of shape (..., sequence_length, d_model)
        """
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv,
            "... seq (three h d) -> three ... h seq d",
            three=3,
            h=self.num_heads,
            d=self.d_head,
        )
        seq_len = x.shape[-2]

        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        out = rearrange(attn, "... h seq d -> ... seq (h d)")
        return self.o_proj(out)
