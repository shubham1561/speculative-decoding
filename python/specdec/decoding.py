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

from .sampling import warp_probs


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
    timeline: list = field(default_factory=list)   # (seconds since start, new tokens so far) per step

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


@torch.no_grad()
def warm_up(lms, seconds=0.3):
    """Keep the GPU busy for a moment so it is at full clock speed before a timed run.

    Laptop GPUs drop to a low clock while idle (e.g. while you type); whichever method
    runs first would otherwise pay for the ramp-up. Does nothing on the CPU.
    """
    lms = [lm for lm in lms if lm.device.type == "cuda"]
    if not lms:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for lm in lms:
            lm.forward([0], lm.new_cache())
        torch.cuda.synchronize()


def _stop_set(eos_id, stop_ids):
    stops = set(stop_ids or ())
    if eos_id is not None:
        stops.add(eos_id)
    return stops


def _prefill(lm, cache, seq):
    if len(seq) > 1:
        lm.forward(seq[:-1], cache)


def _budget(lm, seq_len):
    """How many more tokens fit in this model's context, or None if unlimited."""
    return None if not lm.max_len else lm.max_len - seq_len


@torch.no_grad()
def autoregressive_generate(target, prompt, max_new_tokens, temperature=0.0, top_k=0, top_p=1.0,
                            eos_id=None, generator=None, stop_ids=None):
    """The baseline: one big-model call per token.

    Generation ends after eos_id or after any token in stop_ids (that token is kept).
    Like every streaming decoder (and HF's generate), it reads each token back to the
    CPU once per step, to check for a stop token.
    """
    assert len(prompt) >= 1, "prompt must have at least one token"
    seq = list(prompt)
    stats = GenStats()
    stops = _stop_set(eos_id, stop_ids)
    greedy = temperature <= 0
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
        if greedy:
            t = int(logits.argmax())
        else:
            t = int(torch.multinomial(warp_probs(logits, temperature, top_k, top_p), 1, generator=generator))
        seq.append(t)
        new.append(t)
        stats.timeline.append((time.perf_counter() - t0, len(new)))
        if t in stops:
            break
    _sync(target.device)
    stats.seconds = time.perf_counter() - t0
    stats.new_tokens = len(new)
    return GenResult(new, stats, ["target"] * len(new))


@torch.no_grad()
def speculative_generate(target, draft, prompt, max_new_tokens, gamma=4, temperature=0.0, top_k=0,
                         top_p=1.0, eos_id=None, generator=None, stop_ids=None):
    """Speculative decoding. Same output distribution as autoregressive_generate(target, ...).

    Speed: inside a round every token stays on the GPU -- draft tokens are fed back as
    tensors and accept/reject is one vectorised step -- so the CPU waits for the GPU
    only ONCE per round, when it reads the kept tokens.
    """
    assert len(prompt) >= 1, "prompt must have at least one token"
    assert gamma >= 0
    vocab = min(target.vocab_size, draft.vocab_size)   # HF models sometimes pad the vocab differently
    dev = target.device
    greedy = temperature <= 0
    seq = list(prompt)
    stats = GenStats()
    sources = []
    stops = _stop_set(eos_id, stop_ids)

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
        last = torch.tensor([seq[-1]], dtype=torch.long, device=dev)   # GPU is idle here: cheap

        # 1. DRAFT: g tokens, fed back as GPU tensors (no CPU round trip) ---------
        xs = torch.empty(0, dtype=torch.long, device=dev)
        qs = None
        if g > 0:
            catch_up = seq[draft.cache_len(d_cache):]  # tokens the draft cache has not seen (incl. last)
            logits = draft.forward(catch_up, d_cache)[-1, :vocab]
            stats.draft_calls += 1
            x_list, q_list = [], []
            for i in range(g):
                if greedy:
                    x = logits.argmax(-1, keepdim=True)
                else:
                    q = warp_probs(logits, temperature, top_k, top_p)
                    x = torch.multinomial(q, 1, generator=generator)
                    q_list.append(q)
                x_list.append(x)
                if i < g - 1:
                    logits = draft.forward(x, d_cache)[-1, :vocab]
                    stats.draft_calls += 1
            xs = torch.cat(x_list)
            qs = torch.stack(q_list) if q_list else None

        # 2. VERIFY: one big-model pass over [last, x1..xg] --------------------
        t_logits = target.forward(torch.cat([last, xs]), t_cache)[:, :vocab]
        stats.target_calls += 1

        # 3. ACCEPT / REJECT, all on the GPU ----------------------------------
        if greedy:
            # accept x_i while it equals the target's own choice; then take the target's choice
            choice = t_logits.argmax(-1)                                   # (g+1,)
            n = (choice[:g] == xs).long().cumprod(0).sum()
            new_tok = choice.index_select(0, n.view(1))
        else:
            ps = warp_probs(t_logits, temperature, top_k, top_p)          # (g+1, V)
            if g > 0:
                idx = xs.view(-1, 1)
                px = ps[:g].gather(1, idx).squeeze(1)
                qx = qs.gather(1, idx).squeeze(1)
                u = torch.rand(g, device=ps.device, generator=generator)
                accept = u * qx < px                                       # prob min(1, p/q)
                n = accept.long().cumprod(0).sum()                         # accepted before 1st reject
                res = torch.clamp(ps[:g] - qs, min=0.0)                    # residual max(p-q, 0) ...
                tot = res.sum(-1, keepdim=True)
                res = torch.where(tot > 0, res / tot.clamp_min(1e-30), ps[:g])   # ... normalised
                dists = torch.cat([res, ps[g:]])        # row n: residual if rejected, p_(g+1) if not
            else:
                n = torch.zeros((), dtype=torch.long, device=ps.device)
                dists = ps
            new_tok = torch.multinomial(dists.index_select(0, n.view(1)), 1, generator=generator).view(1)

        # the ONE CPU <-> GPU sync of the round
        out = torch.cat([xs, n.view(1), new_tok]).tolist()
        xs_cpu, n, new_tok = out[:g], out[g], out[g + 1]

        stats.rounds += 1
        stats.proposed += g
        stats.accepted += n
        stats.rejected += int(n < g)
        stats.accepted_per_round.append(n)

        kept = xs_cpu[:n] + [new_tok]
        sources += ["draft"] * n + ["target"]
        seq.extend(kept)

        # 4. ROLLBACK: both caches must hold seq[:-1] (or a prefix of it) --------
        target.crop(t_cache, len(seq) - 1)
        if draft.cache_len(d_cache) > len(seq) - 1:
            draft.crop(d_cache, len(seq) - 1)

        hit = next((i for i, t in enumerate(kept) if t in stops), None)
        if hit is not None:                            # stop exactly where normal decoding would
            cut = len(seq) - len(kept) + hit + 1
            sources = sources[: cut - start_len]
            seq = seq[:cut]
        stats.timeline.append((time.perf_counter() - t0, len(seq) - start_len))
        if hit is not None:
            break

    _sync(dev)
    stats.seconds = time.perf_counter() - t0
    new = seq[start_len:]
    stats.new_tokens = len(new)
    return GenResult(new, stats, sources)
