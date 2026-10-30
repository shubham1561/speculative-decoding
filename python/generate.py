"""See speculative decoding happen, one prompt at a time.

    python generate.py --prompt "Once upon a time, a little dog"
    python generate.py --pair gpt2 --prompt "The history of Rome" --gamma 5 --temperature 0.8
    python generate.py --draft plain        # compare with --draft distilled

Colours in the output:
    green  = guessed by the draft and accepted by the target (came "for free")
    yellow = produced by the target itself (a correction, or the bonus token)
Then it runs normal decoding on the same prompt and prints both speeds.
"""
import argparse
import os

import torch

import config
from specdec.data import get_tokenizer
from specdec.decoding import autoregressive_generate, speculative_generate
from specdec.pairs import load_lm, exists, inference_dtype
from specdec.training import pick_device

GREEN, YELLOW, RESET = "\033[92m", "\033[93m", "\033[0m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=list(config.PAIRS), default="scratch")
    ap.add_argument("--draft", default=None, help="plain / distilled (default: distilled if it exists)")
    ap.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily.")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if os.name == "nt":
        os.system("")                      # turns on colours in the Windows terminal

    dev = pick_device(a.device)
    pair = config.PAIRS[a.pair]
    names = [a.draft] if a.draft else ["distilled", "plain"]
    dname = next((n for n in names if n in pair["drafts"] and exists(pair["drafts"][n])), None)
    if dname is None:
        raise SystemExit(f"no draft found for pair '{a.pair}' -- train it first (see the guides)")
    tok = get_tokenizer(pair["tokenizer"])
    dtype = inference_dtype(dev, pair.get("dtype"))
    target = load_lm(pair["target"], dev, dtype)
    draft = load_lm(pair["drafts"][dname], dev, dtype)
    prompt = tok.encode(a.prompt)
    eos = getattr(tok, "eot", None)
    kw = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, eos_id=eos)

    speculative_generate(target, draft, prompt, 8, a.gamma, **kw)        # warm-up
    g = torch.Generator(device=dev).manual_seed(a.seed)
    spec = speculative_generate(target, draft, prompt, a.max_new_tokens, a.gamma, generator=g, **kw)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    base = autoregressive_generate(target, prompt, a.max_new_tokens, generator=g, **kw)

    print(f"\npair {a.pair}, draft '{dname}', gamma {a.gamma}, temperature {a.temperature}\n")
    out = [a.prompt]
    for t, src in zip(spec.tokens, spec.sources):
        out.append((GREEN if src == "draft" else YELLOW) + tok.decode([t]) + RESET)
    print("".join(out))
    s, b = spec.stats, base.stats
    print(f"\n{GREEN}green{RESET} = draft token accepted, {YELLOW}yellow{RESET} = made by the target\n")
    print(f"speculative: {s.new_tokens} tokens in {s.seconds:.2f} s = {s.tokens_per_second:.1f} tok/s, "
          f"{s.accepted}/{s.proposed} draft tokens accepted ({s.acceptance_rate:.0%}), "
          f"{s.tokens_per_target_call:.2f} tokens per target call")
    print(f"normal:      {b.new_tokens} tokens in {b.seconds:.2f} s = {b.tokens_per_second:.1f} tok/s")
    print(f"speedup:     {s.tokens_per_second / b.tokens_per_second:.2f}x")
    if a.temperature == 0:
        print(f"same text as normal greedy decoding: {'YES' if spec.tokens == base.tokens else 'NO (see GUIDE_4, Problems)'}")


if __name__ == "__main__":
    main()
