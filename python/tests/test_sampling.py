import torch

from specdec.sampling import warp_probs, residual, acceptance_prob


def test_greedy_is_one_hot():
    p = warp_probs(torch.tensor([0.1, 2.0, -1.0]), temperature=0)
    assert p.tolist() == [0.0, 1.0, 0.0]


def test_top_k_and_top_p():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    assert (warp_probs(logits, 1.0, top_k=2) > 0).sum() == 2
    p = warp_probs(logits, 1.0, top_p=0.75)          # 0.5 + 0.3 >= 0.75 -> keep two
    assert (p > 0).sum() == 2
    assert torch.allclose(p[:2], torch.tensor([0.625, 0.375]))
    p = warp_probs(logits, 1.0, top_p=0.4)           # first token alone already >= 0.4
    assert (p > 0).sum() == 1


def test_warping_works_on_batches():
    logits = torch.randn(5, 30)
    p = warp_probs(logits, 0.7, top_k=10, top_p=0.9)
    for i in range(5):
        assert torch.allclose(p[i], warp_probs(logits[i], 0.7, top_k=10, top_p=0.9))
    assert torch.allclose(p.sum(-1), torch.ones(5))


def test_residual_and_exactness_identity():
    """q(x)*min(1,p/q) + P(reject)*residual(x) must equal p(x) for every x."""
    torch.manual_seed(0)
    for _ in range(20):
        p = torch.softmax(torch.randn(12), 0)
        q = torch.softmax(torch.randn(12), 0)
        accept_part = torch.minimum(p, q)
        reject_prob = 1 - accept_part.sum()
        total = accept_part + reject_prob * residual(p, q)
        assert torch.allclose(total, p, atol=1e-6)
        assert abs(acceptance_prob(p, q) - (1 - 0.5 * (p - q).abs().sum())) < 1e-6
