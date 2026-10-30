"""Turning logits into the probability distribution we sample from.

The SAME warping (temperature, top-k, top-p) is applied to the target and the
draft. Speculative decoding then reproduces the *warped* target distribution
exactly -- which is the distribution normal sampling would have used.
"""
import torch


def warp_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """logits (..., V) -> probabilities (..., V), in float32.

    temperature == 0 means greedy: a one-hot vector on the argmax.
    """
    logits = logits.float()
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs

    logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        # drop a token if the tokens *before* it already reach top_p (the first token is always kept)
        drop_sorted = (sorted_probs.cumsum(-1) - sorted_probs) >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return torch.softmax(logits, dim=-1)


def sample(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """Draw one token id from a 1-D probability vector."""
    return int(torch.multinomial(probs, 1, generator=generator).item())


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The distribution to resample from after a rejection: norm(max(p - q, 0)).

    If p == q exactly, rejection is impossible, but return p to stay safe.
    """
    r = torch.clamp(p - q, min=0.0)
    s = r.sum()
    if s <= 0:
        return p
    return r / s


def acceptance_prob(p: torch.Tensor, q: torch.Tensor) -> float:
    """Expected acceptance of one draft token: sum_x min(p(x), q(x)) = 1 - TV(p, q)."""
    return float(torch.minimum(p, q).sum())
