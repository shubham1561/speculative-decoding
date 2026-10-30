"""The two ways to generate text: normal (autoregressive) and speculative.

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023), one round:

  1. DRAFT   the small model proposes `gamma` tokens x1..xg, one at a time,
             remembering the distribution q_i each was sampled from.
  2. VERIFY  the big model reads [last token, x1..xg] in ONE forward pass and
             gives its distributions p_1 .. p_(g+1).
  3. ACCEPT  x_i with probability min(1, p_i(x_i) / q_i(x_i)), left to right.
             At the first rejection, sample a replacement from norm(max(p_i - q_i, 0))
             and stop. If all g are accepted, sample a BONUS token from p_(g+1).
  4. ROLLBACK the KV caches to the tokens that were actually kept.

Every round yields between 1 and gamma+1 tokens for one big-model call, and the
output follows the big model's distribution exactly (proof: docs/THEORY.md).

Cache rule used here: both caches hold the committed sequence EXCEPT its last
token. The last token is fed again at the start of the next round.
"""
import time
from dataclasses import dataclass, field

import torch

from .sampling import warp_probs, sample, residual


@dataclass
class GenStats:
    new_tokens: int = 0
    target_calls: int = 0          # big-model forward passes (prefill not counted)
    draft_calls: int = 0           # small-model forward passes (prefill not counted)
    rounds: int = 0
    proposed: int = 0              # draft tokens offered for verification
    accepted: int = 0              # draft tokens kept
    rejected: int = 0              # rounds that ended with a rejection
    accepted_per_round: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def alpha(self):
        """Per-token acceptance estimate (accept/reject are Bernoulli trials until the first reject)."""
        n = self.accepted + self.rejected
        return self.accepted / n if n else float("nan")

    @property
    def tokens_per_target_call(self):
        return self.new_tokens / self.target_calls if self.target_calls else float("nan")

    @property
    def tokens_per_second(self):
        return self.new_tokens / self.seconds if self.seconds else float("nan")


@dataclass
class GenResult:
    tokens: list                   # the NEW tokens only
    stats: GenStats
    sources: list                  # per new token: "draft" (accepted proposal) or "target"


def _sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _prefill(lm, cache, seq):
    if len(seq) > 1:
        lm.forward(seq[:-1], cache)


def _budget(lm, seq_len):
    """How many more tokens fit in this model's context, or None if unlimited."""
    return None if not lm.max_len else lm.max_len - seq_len


@torch.no_grad()
def autoregressive_generate(target, prompt, max_new_tokens, temperature=0.0, top_k=0, top_p=1.0,
                            eos_id=None, generator=None):
    """The baseline: one big-model call per token."""
    assert len(prompt) >= 1, "prompt must have at least one token"
    seq = list(prompt)
    stats = GenStats()
    _sync(target.device)
    t0 = time.perf_counter()
    cache = target.new_cache()
    _prefill(target, cache, seq)
    limit = _budget(target, len(seq))
    if limit is not None:
        max_new_tokens = min(max_new_tokens, limit + 1)
    new = []
    while len(new) < max_new_tokens:
        logits = target.forward([seq[-1]], cache)[-1]
        stats.target_calls += 1
        t = sample(warp_probs(logits, temperature, top_k, top_p), generator)
        seq.append(t)
        new.append(t)
        if eos_id is not None and t == eos_id:
            break
    _sync(target.device)
    stats.seconds = time.perf_counter() - t0
    stats.new_tokens = len(new)
    return GenResult(new, stats, ["target"] * len(new))


@torch.no_grad()
def speculative_generate(target, draft, prompt, max_new_tokens, gamma=4, temperature=0.0, top_k=0,
                         top_p=1.0, eos_id=None, generator=None):
    """Speculative decoding. Same output distribution as autoregressive_generate(target, ...)."""
    assert len(prompt) >= 1, "prompt must have at least one token"
    assert gamma >= 0
    vocab = min(target.vocab_size, draft.vocab_size)   # HF models sometimes pad the vocab differently
    dev = target.device
    seq = list(prompt)
    stats = GenStats()
    sources = []

    _sync(dev)
    t0 = time.perf_counter()
    t_cache, d_cache = target.new_cache(), draft.new_cache()
    _prefill(target, t_cache, seq)
    _prefill(draft, d_cache, seq)
    start_len = len(seq)

    while (produced := len(seq) - start_len) < max_new_tokens:
        remaining = max_new_tokens - produced
        g = min(gamma, remaining - 1)                  # a round makes up to g+1 tokens
        for lm in (target, draft):                     # stay inside each model's context window
            room = _budget(lm, len(seq))
            if room is not None:
                g = min(g, room)
        if g < 0:
            break

        # 1. DRAFT ---------------------------------------------------------
        xs, qs = [], []
        if g > 0:
            catch_up = seq[draft.cache_len(d_cache):]  # tokens the draft cache has not seen (incl. last)
            logits = draft.forward(catch_up, d_cache)[-1]
            stats.draft_calls += 1
            for i in range(g):
                q = warp_probs(logits[:vocab], temperature, top_k, top_p)
                x = sample(q, generator)
                xs.append(x)
                qs.append(q)
                if i < g - 1:
                    logits = draft.forward([x], d_cache)[-1]
                    stats.draft_calls += 1

        # 2. VERIFY: one big-model pass over [last, x1..xg] --------------------
        t_logits = target.forward([seq[-1]] + xs, t_cache)
        stats.target_calls += 1
        ps = warp_probs(t_logits[:, :vocab], temperature, top_k, top_p)

        # 3. ACCEPT / REJECT -------------------------------------------------
        n = 0
        new_tok = None
        for i in range(g):
            x = xs[i]
            u = torch.rand((), device=ps.device, generator=generator)
            if u * qs[i][x] < ps[i][x]:                # accept with prob min(1, p/q)
                n += 1
            else:
                new_tok = sample(residual(ps[i], qs[i]), generator)
                break
        if new_tok is None:                            # all accepted -> bonus token for free
            new_tok = sample(ps[g], generator)

        stats.rounds += 1
        stats.proposed += g
        stats.accepted += n
        stats.rejected += int(n < g)
        stats.accepted_per_round.append(n)

        kept = xs[:n] + [new_tok]
        sources += ["draft"] * n + ["target"]
        seq.extend(kept)

        # 4. ROLLBACK: both caches must hold seq[:-1] (or a prefix of it) --------
        target.crop(t_cache, len(seq) - 1)
        if draft.cache_len(d_cache) > len(seq) - 1:
            draft.crop(d_cache, len(seq) - 1)

        if eos_id is not None and eos_id in kept:
            cut = len(seq) - len(kept) + kept.index(eos_id) + 1
            sources = sources[: cut - start_len]
            seq = seq[:cut]
            break

    _sync(dev)
    stats.seconds = time.perf_counter() - t0
    new = seq[start_len:]
    stats.new_tokens = len(new)
    return GenResult(new, stats, sources)
