"""Check the whole machinery in a few minutes, on CPU or GPU, with no downloads.

    python smoke_test.py

It makes up a small set of simple stories, then runs every step of the real
pipeline with tiny models: prepare data -> train target -> train draft ->
distill draft -> benchmark. Everything goes to a temporary folder; your real
data/, checkpoints/ and results/ are NOT touched.

It proves the code works. The tiny models write nonsense -- that is expected.
"""
import os
import random
import sys
import tempfile
import time

import torch

import prepare_data
import train_scratch
import distill_draft
from specdec.benchmarking import run_benchmark
from specdec.data import get_tokenizer, story_prompts, read_meta
from specdec.training import pick_device

NAMES = ["Lily", "Tom", "Mia", "Ben", "Sara", "Max", "Anna", "Sam"]
ANIMALS = ["dog", "cat", "bird", "fish", "frog", "duck"]
PLACES = ["park", "garden", "forest", "beach", "house", "school"]
FEELINGS = ["happy", "sad", "excited", "tired", "brave", "kind"]


def make_story(rng):
    n, a, p, f = rng.choice(NAMES), rng.choice(ANIMALS), rng.choice(PLACES), rng.choice(FEELINGS)
    return (f"Once upon a time, there was a little {a} named {n}. {n} liked to play in the {p}. "
            f"One day, {n} felt very {f}. {n} went to the {p} with a friend. "
            f"They played all day and had fun. The end.")


def main():
    t0 = time.time()
    dev = pick_device("auto")
    print(f"PyTorch {torch.__version__}, device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "  (no GPU found -- fine for this test)"))
    rng = random.Random(0)
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    with tempfile.TemporaryDirectory() as tmp:
        data, ckpt, logs, res = (os.path.join(tmp, d) for d in ("data", "ckpt", "logs", "results"))
        quiet = lambda *a, **k: None

        print("\n1/5 preparing made-up stories (byte tokenizer, no download)")
        prepare_data.prepare_from_lists([make_story(rng) for _ in range(3000)],
                                        [make_story(rng) for _ in range(100)], "bytes", data, "smoke test")
        check("data files written", os.path.exists(os.path.join(data, "train.bin")))

        print("\n2/5 training a tiny target")
        _, tl = train_scratch.run("target", "tiny", data, ckpt, logs, device=str(dev), log=quiet)
        check("target trains", tl < 2.0, f"(val loss {tl:.2f}, random would be {torch.log(torch.tensor(257.)):.2f})")

        print("3/5 training a tiny draft")
        _, dl = train_scratch.run("draft", "tiny", data, ckpt, logs, device=str(dev), log=quiet)
        check("draft trains", dl < 3.0, f"(val loss {dl:.2f})")

        print("4/5 distilling the draft from the target")
        _, before, _ = distill_draft.run("scratch", "tiny", "kl", data, ckpt, logs, device=str(dev), log=quiet)
        ck = torch.load(os.path.join(ckpt, "draft_distilled.pt"), weights_only=False)
        check("distillation runs and saves", ck is not None, f"(acceptance before: {before['val_accept']:.2f})")

        print("5/5 benchmark: normal vs speculative decoding")
        pair = {"tokenizer": "bytes", "target": ("own", os.path.join(ckpt, "target.pt")),
                "drafts": {"plain": ("own", os.path.join(ckpt, "draft.pt")),
                           "distilled": ("own", os.path.join(ckpt, "draft_distilled.pt"))}}
        tok = get_tokenizer("bytes")
        prompts = story_prompts(data, 4, 16, read_meta(data)["eot"])
        rows, _ = run_benchmark(pair_name="smoke", pair=pair, tokenizer=tok, prompts=prompts, device=dev,
                                out_dir=res, gammas=[1, 2, 4], temperatures=[0.0, 0.8], max_new_tokens=48,
                                prompt_set="stories", log=quiet)
        greedy = [r for r in rows if r["temperature"] == 0]
        exact = min(r["greedy_exact_match"] for r in greedy)
        check("greedy speculative == greedy normal", exact == 1.0, f"({exact:.0%} of prompts identical)")
        acc = max(r["accept_rate"] for r in rows)
        check("draft tokens get accepted", acc > 0.2, f"(best acceptance {acc:.0%})")
        files = os.listdir(res)
        check("results table and plot written", "RESULTS.md" in files and any(f.endswith(".png") for f in files))

    ok = all(r[1] for r in results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f} s")
    if ok:
        print("Next: python prepare_data.py   (GUIDE_2, step 1)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
