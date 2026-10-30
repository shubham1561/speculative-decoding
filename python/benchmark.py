"""Step 4: measure the speedup.

    python benchmark.py --pair scratch                    # your own target + drafts
    python benchmark.py --pair gpt2                       # gpt2-large + distilgpt2 (plain and distilled)
    python benchmark.py --pair gpt2 --prompts general     # any-topic prompts instead of stories
    python benchmark.py --pair qwen --prompts general     # Qwen2.5-1.5B + 0.5B (inference only)
    python benchmark.py --pair scratch --gammas 2 4 --temperatures 0 --n_prompts 5   # quick

For every draft, temperature and gamma it runs the same prompts with normal
decoding and with speculative decoding, and writes to results/:
    <pair>_<prompts>_<date>.json / .csv   every number
    <pair>_<prompts>_<date>.png           speedup and acceptance vs gamma
    RESULTS.md                            summary table (copy it into the README)
"""
import argparse

import config
from specdec.benchmarking import run_benchmark
from specdec.data import get_tokenizer, read_meta, story_prompts
from specdec.pairs import inference_dtype
from specdec.training import pick_device


def build_prompts(kind, tokenizer, n, prompt_tokens, data_dir=config.DATA_DIR, seed=0):
    if kind == "general":
        texts = config.GENERAL_PROMPTS[:n]
        return [tokenizer.encode(t) for t in texts]
    meta = read_meta(data_dir)
    data_tok = get_tokenizer(meta["tokenizer"])
    raw = story_prompts(data_dir, n, prompt_tokens, meta["eot"], seed)
    if data_tok.name == tokenizer.name:
        return raw
    return [tokenizer.encode(data_tok.decode(p)) for p in raw]       # e.g. stories for the Qwen pair


def main():
    b = config.BENCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=list(config.PAIRS), default="scratch")
    ap.add_argument("--prompts", choices=["stories", "general"], default="stories")
    ap.add_argument("--drafts", nargs="*", default=None, help="e.g. plain distilled (default: all that exist)")
    ap.add_argument("--gammas", nargs="+", type=int, default=b["gammas"])
    ap.add_argument("--temperatures", nargs="+", type=float, default=b["temperatures"])
    ap.add_argument("--n_prompts", type=int, default=b["n_prompts"])
    ap.add_argument("--max_new_tokens", type=int, default=b["max_new_tokens"])
    ap.add_argument("--top_k", type=int, default=b["top_k"])
    ap.add_argument("--top_p", type=float, default=b["top_p"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=config.RESULTS_DIR)
    a = ap.parse_args()

    pair = config.PAIRS[a.pair]
    tok = get_tokenizer(pair["tokenizer"])
    prompts = build_prompts(a.prompts, tok, a.n_prompts, b["prompt_tokens"], seed=b["seed"])
    dev = pick_device(a.device)
    run_benchmark(pair_name=a.pair, pair=pair, tokenizer=tok, prompts=prompts, device=dev,
                  dtype=inference_dtype(dev, pair.get("dtype")),
                  out_dir=a.out, gammas=a.gammas, temperatures=a.temperatures, max_new_tokens=a.max_new_tokens,
                  top_k=a.top_k, top_p=a.top_p, seed=b["seed"], prompt_set=a.prompts, draft_names=a.drafts)


if __name__ == "__main__":
    main()
