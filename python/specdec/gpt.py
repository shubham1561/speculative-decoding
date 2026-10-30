"""A small GPT written from scratch, with its own KV cache.

The cache is pre-allocated, so "rolling back" rejected draft tokens is just
moving a length pointer (`KVCache.crop`) -- no copies, no re-computation.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256      # longest sequence the model can see (prompt + generated)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False

    def to_dict(self):
        return asdict(self)


class KVCache:
    """Keys/values for every layer, pre-allocated to `max_len` positions.

    Only the first `length` positions are valid. crop(n) keeps the first n.
    """

    def __init__(self, cfg: GPTConfig, device, dtype, batch: int = 1, max_len: int | None = None):
        self.max_len = max_len or cfg.block_size
        head_dim = cfg.n_embd // cfg.n_head
        shape = (batch, cfg.n_head, self.max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.length = 0

    def crop(self, n: int):
        if not 0 <= n <= self.length:
            raise ValueError(f"cannot crop cache of length {self.length} to {n}")
        self.length = n


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache: KVCache | None = None, layer: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if cache is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            start = cache.length
            end = start + T
            cache.k[layer][:B, :, start:end] = k
            cache.v[layer][:B, :, start:end] = v
            k_all = cache.k[layer][:B, :, :end]
            v_all = cache.v[layer][:B, :, :end]
            if T == 1:
                mask = None                     # one new token may look at everything
            else:
                # new token i (absolute position start+i) may look at positions <= start+i
                rows = torch.arange(T, device=x.device).unsqueeze(1) + start
                cols = torch.arange(end, device=x.device).unsqueeze(0)
                mask = cols <= rows
            y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None, layer=0):
        x = x + self.attn(self.ln1(x), cache, layer)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight          # weight tying (saves ~vocab*n_embd params)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():        # GPT-2 style scaled init of residual projections
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel() if non_embedding else n

    def new_cache(self, device=None, dtype=None, batch=1) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg, device or p.device, dtype or p.dtype, batch)

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.block_size:
            raise ValueError(f"sequence length {start + T} > block_size {self.cfg.block_size}")
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        if cache is not None:
            cache.length = start + T
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1)
        return logits, loss


def load_gpt(path, device="cpu", dtype=None) -> GPT:
    """Load a checkpoint written by the trainer (`*.pt`)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    model.eval()
    model.tokenizer_name = ck.get("tokenizer", "gpt2")
    return model
