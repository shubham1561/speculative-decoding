"""Correctness of speculative decoding.

1. Greedy (temperature 0): output must be token-for-token the same as the
   target alone, for any draft (even a random one) and any gamma.
2. Sampling: the distribution of outputs must match the target's exactly.
   Checked by sampling thousands of times and comparing with the exact
   probabilities (total variation distance).
"""

import pytest
import torch

from specdec.adapters import OwnGPT, HFModel
from specdec.decoding import autoregressive_generate, speculative_generate
from specdec.gpt import GPT, GPTConfig
from specdec.sampling import warp_probs

def own(seed, vocab=60, layers=2, embd=32, block=64):
    torch.manual_seed(seed)
    return OwnGPT(GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=layers, n_head=2, n_embd=embd)), f"m{seed}")

@pytest.mark.parametrize("gamma", [0, 1, 2, 4, 7])
def test_greedy_identical_to_target(gamma):
    target, draft = own(0, layers=3), own(1, layers=1)       # unrelated random draft
    for p in ([5], [3, 9, 1, 4], list(range(10))):
        base = autoregressive_generate(target, p, 30)
        spec = speculative_generate(target, draft, p, 30, gamma=gamma)
        assert spec.tokens == base.tokens
        assert len(spec.tokens) == 30

def test_greedy_with_draft_equal_to_target_accepts_everything():
    target = own(0)
    spec = speculative_generate(target, target, [1, 2, 3], 25, gamma=4)
    assert spec.stats.acceptance_rate == 1.0
    assert spec.tokens == autoregressive_generate(target, [1, 2, 3], 25).tokens

def test_context_window_and_budget():
    target, draft = own(0, block=20), own(1, block=20)
    prompt = list(range(15))
    base = autoregressive_generate(target, prompt, 50)
    spec = speculative_generate(target, draft, prompt, 50, gamma=4)
    assert len(base.tokens) == 6                              # 20 - 15 + 1
    assert spec.tokens == base.tokens

def test_eos_stops_generation():
    target, draft = own(0), own(1)
    base = autoregressive_generate(target, [4, 5], 40)
    eos = base.tokens[10]
    spec = speculative_generate(target, draft, [4, 5], 40, gamma=3, eos_id=eos)
    first = base.tokens.index(eos)
    assert spec.tokens == base.tokens[: first + 1]

def test_huggingface_model_adapter():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg_t = transformers.GPT2Config(vocab_size=70, n_positions=64, n_embd=32, n_layer=3, n_head=2,
                                    bos_token_id=0, eos_token_id=0)
    cfg_d = transformers.GPT2Config(vocab_size=70, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                                    bos_token_id=0, eos_token_id=0)
    target = HFModel(transformers.GPT2LMHeadModel(cfg_t))
    draft = HFModel(transformers.GPT2LMHeadModel(cfg_d))
    for gamma in (1, 3, 5):
        base = autoregressive_generate(target, [1, 2, 3], 25)
        spec = speculative_generate(target, draft, [1, 2, 3], 25, gamma=gamma)
        assert spec.tokens == base.tokens

def _exact_two_token_distribution(target, prompt, **warp):
    p1 = warp_probs(target.forward(prompt, target.new_cache())[-1], **warp)
    V = p1.numel()
    joint = torch.zeros(V, V)
    for a in range(V):
        if p1[a] > 0:
            p2 = warp_probs(target.forward(prompt + [a], target.new_cache())[-1], **warp)
            joint[a] = p1[a] * p2
    return joint

@pytest.mark.parametrize("warp", [dict(temperature=1.0), dict(temperature=0.7, top_p=0.9)])
def test_sampling_matches_target_distribution(warp):
    # sharp-ish random models over 8 tokens, draft deliberately different
    torch.manual_seed(0)
    target = own(3, vocab=8, embd=32)
    draft = own(4, vocab=8, embd=16)
    for lm in (target, draft):
        with torch.no_grad():
            lm.model.wte.weight.mul_(8)
    prompt = [1, 2]
    exact = _exact_two_token_distribution(target, prompt, top_k=0, **{"top_p": 1.0, **warp})
    n = 4000
    counts = torch.zeros(8, 8)
    g = torch.Generator().manual_seed(123)
    accepted = 0
    for _ in range(n):
        r = speculative_generate(target, draft, prompt, 2, gamma=3, generator=g, top_k=0,
                                 **{"top_p": 1.0, **warp})
        counts[r.tokens[0], r.tokens[1]] += 1
        accepted += r.stats.accepted
    tv = 0.5 * (counts / n - exact).abs().sum().item()
    assert accepted > 0, "draft never accepted -- test would not exercise the rejection path"
    assert tv < 0.06, f"total variation {tv:.3f} too large"


def test_stop_tokens_end_both_methods_at_the_same_place():
    target, draft = own(0), own(1)
    base = autoregressive_generate(target, [7, 8], 40)
    stop = {base.tokens[12], base.tokens[20]}
    first = min(i for i, t in enumerate(base.tokens) if t in stop)
    for gamma in (1, 3, 6):
        spec = speculative_generate(target, draft, [7, 8], 40, gamma=gamma, stop_ids=stop)
        ar = autoregressive_generate(target, [7, 8], 40, stop_ids=stop)
        assert spec.tokens == ar.tokens == base.tokens[: first + 1]
        assert len(spec.sources) == len(spec.tokens)
