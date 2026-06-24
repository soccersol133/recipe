"""
Ralph-base — minimal Llama-style decoder-only transformer.

Patchable surface for the launch track. Miners may modify any of the modules
here (attention variant, normalization, activation, etc.) as part of a
submitted recipe patch. The training loop in `recipe/train.py` instantiates
this model from a config.

Defaults target ~50M parameters at dim=512, n_layers=8 — small enough for
fast Phase 0 iteration, large enough to produce meaningful val_bpb gradients
under canonical training. Configs in `configs/` override these for proxy /
confirmation / scale proof-test variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RalphConfig:
    vocab_size: int = 50257  # GPT-2 BPE
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    ffn_mult: float = 8 / 3  # Llama-style
    max_seq_len: int = 1024
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x_f32 = x.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(var + eps)
    return (x_normed * weight).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float, device: torch.device) -> torch.Tensor:
    """Returns a tensor of shape (max_seq_len, head_dim // 2, 2) of cos, sin pairs."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.stack([cos, sin], dim=-1)  # (max_seq_len, head_dim // 2, 2)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings. x is (batch, n_heads, seq, head_dim). rope_cache is
    (seq, head_dim // 2, 2). Returns same shape as x.
    """
    seq = x.shape[-2]
    cos = rope_cache[:seq, :, 0]  # (seq, head_dim // 2)
    sin = rope_cache[:seq, :, 1]
    # split last dim into pairs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # rotate
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: RalphConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dim = cfg.dim
        assert cfg.dim == cfg.n_heads * cfg.head_dim, "dim must equal n_heads * head_dim"
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        # Mark as residual-path output for depth-scaled init (GPT-2 §2.3).
        self.out_proj._is_residual_out = True
        # QK-norm: per-head RMSNorm on queries and keys before RoPE. Bounds the
        # attention-logit scale so it can't drift, which is especially important
        # under the Muon optimizer's aggressive orthogonalized updates (see
        # recipe/train.py). Strong synergy with Muon; standard in modern speedruns.
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.dim, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)  # QK-norm (per head_dim, before RoPE)
        k = self.k_norm(k)
        q = apply_rope(q, rope_cache)
        k = apply_rope(k, rope_cache)
        # Causal self-attention via SDPA (uses flash on supported hardware).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: RalphConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        # Round to multiple of 64 for kernel friendliness.
        hidden = 64 * ((hidden + 63) // 64)
        self.w_gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.dim, bias=False)
        # Mark as residual-path output for depth-scaled init (GPT-2 §2.3).
        self.w_down._is_residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: RalphConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RalphBase(nn.Module):
    """
    Minimal Llama-style decoder-only transformer.
    Inputs: token ids (B, T). Outputs: logits (B, T, vocab_size).
    """

    def __init__(self, cfg: RalphConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base, torch.device("cpu")),
            persistent=False,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.cfg.init_std
            # Scale residual-path output projections by 1/sqrt(2 * n_layers) so
            # that residual stream variance stays ~constant at init (GPT-2 §2.3).
            if getattr(module, "_is_residual_out", False):
                std = std / math.sqrt(2 * self.cfg.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_embed.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert idx.shape[-1] <= self.cfg.max_seq_len, f"sequence {idx.shape[-1]} exceeds max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_embed(idx)
        for block in self.blocks:
            x = block(x, self.rope_cache)
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_embed.weight)
        else:
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Back-compat aliases (rebrand karpa -> ralph, 2026-06).
# The classes were renamed KarpaBase -> RalphBase / KarpaConfig -> RalphConfig.
# These aliases keep `from model import KarpaBase, KarpaConfig` resolving for
# any unmigrated importer or out-of-tree tooling. Checkpoints are unaffected:
# torch.save stores asdict(cfg) under "config" (field names, not the class
# name) and state_dict keys are module-attribute paths, so neither the class
# name nor "Karpa"/"Ralph" is ever serialized. Safe to remove once all
# external consumers cut over.
KarpaConfig = RalphConfig
KarpaBase = RalphBase
