# Theory

Notation: `p(·)` is the target's next-token distribution, `q(·)` the draft's,
both **after** the same temperature / top-k / top-p warping. γ is the number of
draft tokens per round.

## 1. Why one token is exactly right

The draft samples `x ~ q`. The target accepts it with probability
`min(1, p(x)/q(x))`. If rejected, a new token is drawn from

```
r(x) = max(p(x) − q(x), 0) / Σ_y max(p(y) − q(y), 0)
```

Probability the final token is `x`:

```
P(x) = q(x)·min(1, p(x)/q(x))  +  P(reject)·r(x)
     = min(q(x), p(x))         +  (1 − Σ_y min(p(y), q(y))) · r(x)
```

Since `Σ_y max(p(y) − q(y), 0) = 1 − Σ_y min(p(y), q(y))`, the second term is
exactly `max(p(x) − q(x), 0)`, and

```
P(x) = min(p(x), q(x)) + max(p(x) − q(x), 0) = p(x).
```

So the token follows `p` — for **any** draft `q`. A bad draft only costs speed.

`tests/test_sampling.py::test_residual_and_exactness_identity` checks this
identity numerically; `tests/test_speculative.py::test_sampling_matches_target_distribution`
checks the whole multi-token procedure by sampling.

## 2. Why a whole round is exactly right

Within a round, token *i* is only examined if tokens 1…i−1 were accepted, and
then its context is exactly the accepted prefix — the same context normal
decoding would have. Applying section 1 position by position gives the target's
distribution for every token kept. The bonus token after γ acceptances is drawn
from `p_{γ+1}` directly. By induction over rounds, the whole sequence follows
the target's distribution.

**Greedy** (temperature 0) is the special case where `p` and `q` are one-hot:
accept iff the draft's argmax equals the target's argmax; on rejection
`r` is one-hot on the target's argmax. Output is token-for-token identical.

## 3. Acceptance rate

The probability of accepting one draft token is

```
α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)
```

where TV is the total variation distance. This is why `distill_draft.py
--loss tv` exists: minimising TV maximises α directly (DistillSpec, Zhou et al.
2024). `val_accept` in the distillation log is this α, averaged over validation
tokens at temperature 1.

## 4. Expected tokens per round and speedup

Treating each acceptance as an independent coin with probability α, the number
of tokens produced per round (accepted drafts + the correction or bonus) is

```
E[tokens] = 1 + α + α² + … + α^γ = (1 − α^{γ+1}) / (1 − α)
```

One round costs γ draft passes plus one target pass. With
`c = time(draft pass) / time(target pass)`:

```
S(γ) = E[tokens] / (γc + 1) = (1 − α^{γ+1}) / ((1 − α)(γc + 1))
```

This is Theorem 3.8 of Leviathan et al. (2023). It assumes the verify pass over
γ+1 tokens costs the same as a 1-token pass — true while decoding is limited by
memory bandwidth (reading the weights), which is the case at batch size 1.

`benchmark.py` estimates α as `accepted / (accepted + rejections)` (the
maximum-likelihood estimate for trials that stop at the first failure) and c by
timing single-token passes, and plots `S(γ)` next to the measured speedup.

## 5. The optimal γ

`S(γ)` rises while the extra expected tokens are worth more than the extra
draft cost, then falls. Small c and large α push the best γ up. For
α = 0.8, c = 0.2: S(1) = 1.50, S(3) = 1.84, S(4) = 1.87, S(6) = 1.80 — best γ = 4.

## 6. Implementation notes

- **Cache rule.** Both KV caches always hold the committed sequence except its
  last token. Each round feeds `[last, x1 … xγ]` to the target. After the round,
  the target cache is cropped to the new sequence minus its last token; the
  draft cache is cropped the same way, and any tokens it has not seen yet are
  fed in front of the next round's first draft call.
- **One CPU–GPU sync per round.** Small models are limited by overhead, not maths: a
  51M-parameter pass takes ~6.6 ms on a laptop GPU, and every time Python reads a value
  back from the GPU (`.item()`, an `if` on a GPU tensor) the GPU queue drains and waits.
  The first version did that twice per draft token, which cancelled the whole gain
  (1.00× measured vs 1.47× theory). Now draft tokens are fed back to the draft as GPU
  tensors, accept/reject is one vectorised step (`cumprod` of the accept flags gives the
  number accepted; the residual rows are built for all positions and row *n* is
  sampled), and the round's tokens are read back with a single `.tolist()`. Greedy
  decoding skips softmax entirely (argmax only). The baseline reads one token per step,
  as any streaming decoder (and HF's `generate`) does to check for a stop token.
- **Precision.** fp16 can flip a near-tie between two tokens, because a 1-token pass and a
  (γ+1)-token pass use different kernels. Your own small models run in fp32 (`"dtype":
  "float32"` in `config.PAIRS`); it costs almost nothing at this size and keeps greedy
  output identical. GPT-2 large stays in fp16 for memory and speed.
- **Own KV cache** (`specdec/gpt.py`): pre-allocated per layer, so a rollback is
  a pointer move. Multi-token passes on top of a cache use an explicit causal
  mask (`key position ≤ query position`).
- **HuggingFace cache**: `DynamicCache.crop` with a negative length, which works
  in transformers 4.x and 5.x.
- **Vocabulary**: when two HuggingFace models pad their output layer to
  different sizes, logits are cut to the shared size.
- **Budget**: a round never drafts past `max_new_tokens` or the context window,
  so the output length is exactly the same as normal decoding.
