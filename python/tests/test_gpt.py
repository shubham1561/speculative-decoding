"""The KV cache must give exactly the same logits as running the full sequence."""
import torch

from specdec.gpt import GPT, GPTConfig


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50, block_size=64, n_layer=2, n_head=2, n_embd=32)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg)).eval()


def test_cache_matches_full_forward():
    m = tiny()
    ids = torch.randint(0, 50, (1, 20))
    full, _ = m(ids)
    cache = m.new_cache()
    a, _ = m(ids[:, :7], cache=cache)          # several tokens at once
    b, _ = m(ids[:, 7:8], cache=cache)          # one token
    c, _ = m(ids[:, 8:], cache=cache)           # several again
    inc = torch.cat([a, b, c], dim=1)
    assert torch.allclose(full, inc, atol=1e-5)


def test_crop_rolls_back():
    m = tiny()
    ids = torch.randint(0, 50, (1, 12))
    cache = m.new_cache()
    m(ids[:, :8], cache=cache)
    m(torch.randint(0, 50, (1, 3)), cache=cache)   # tokens that get "rejected"
    cache.crop(8)
    after, _ = m(ids[:, 8:], cache=cache)
    full, _ = m(ids)
    assert torch.allclose(full[:, 8:], after, atol=1e-5)


def test_training_reduces_loss():
    m = tiny().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 50, (4, 16))
    first = None
    for _ in range(30):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7
