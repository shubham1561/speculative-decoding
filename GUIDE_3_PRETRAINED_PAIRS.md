# Guide 3 — Pretrained pairs (GPT-2, Qwen)

Your own models (Guide 2) show the method end to end. Real models show it
at scale. The **same** speculative decoding code runs on HuggingFace models —
only the weights are loaded from HuggingFace; the draft/verify/accept loop
and the cache rollback are this repo's own.

| Pair | Target | Draft | Size on disk | Memory (fp16) | Distill on 6 GB? |
|------|--------|-------|-------------:|--------------:|:----------------:|
| `gpt2` | gpt2-large, 774M | distilgpt2, 82M | 3.6 GB | ~2 GB | yes |
| `qwen` | Qwen2.5-1.5B | Qwen2.5-0.5B | 4 GB | ~4 GB | no (inference only) |

Models are downloaded the first time and kept in your HuggingFace cache
(`C:\Users\<you>\.cache\huggingface`), not in the repo.

---

## 1. GPT-2 pair, plain

```bat
python benchmark.py --pair gpt2 --drafts plain --prompts general
```

This measures the stock DistilGPT-2 as a draft for GPT-2 large on any-topic prompts.

---

## 2. Distill DistilGPT-2 towards GPT-2 large (45–90 minutes)

You need the TinyStories data from Guide 2, step 1 (it uses the same GPT-2 tokens).

```bat
python distill_draft.py --pair gpt2
```

GPT-2 large is the teacher (frozen, fp16). DistilGPT-2 learns to copy its
probabilities **on stories**. Result: `checkpoints\distilgpt2_distilled\`
(a normal HuggingFace folder). Watch `val_accept` go up, as in Guide 2, step 4.

Memory: about 2 GB for the teacher, 1.5 GB for the student and its optimiser,
plus the batch. If it runs out, lower `micro_batch` in `config.py` → `distill_hf`.

---

## 3. The experiment: does a domain-distilled draft help only in its domain?

```bat
python benchmark.py --pair gpt2 --prompts stories
python benchmark.py --pair gpt2 --prompts general
```

Each run compares `plain` and `distilled` drafts. Expected pattern:

| Prompts | plain draft | distilled draft |
|---------|-------------|-----------------|
| stories (its training domain) | baseline | **clearly higher acceptance and speedup** |
| general (not its domain) | baseline | about the same, maybe a little lower |

Whatever you find is a result worth writing down — especially if it is *not* this pattern.

---

## 4. Qwen pair (optional, inference only)

```bat
python benchmark.py --pair qwen --prompts general
```

Downloads ~4 GB once. Both models in fp16 fit in 6 GB together for
generation. Training a 0.5B draft would need ~8 GB for the optimiser alone, so
there is no distillation for this pair on 6 GB.

---

## 5. Add your own pair

Any two HuggingFace causal language models **with the same tokenizer** work.
Add them to `PAIRS` in `config.py`:

```python
"pythia": {
    "tokenizer": "EleutherAI/pythia-1.4b",
    "target": ("hf", "EleutherAI/pythia-1.4b"),
    "drafts": {"plain": ("hf", "EleutherAI/pythia-160m")},
},
```

Then `python benchmark.py --pair pythia --prompts general`.

---

## Problems

| What you see | What to do |
|--------------|-----------|
| download stops or is very slow | run it again — HuggingFace resumes downloads |
| `CUDA out of memory` in the benchmark | close other GPU programs. For Qwen, run with nothing else open |
| `skipping draft 'distilled' ... not found` | run `distill_draft.py --pair gpt2` first, or ignore it |
| warnings about `generation_config` / `attention_mask` | harmless — the code feeds one sequence at a time and does not use HF's `generate` |
