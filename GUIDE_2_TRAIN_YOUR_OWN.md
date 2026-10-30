# Guide 2 — Train your own target and draft

You train two GPT models from zero on **TinyStories** (2.1 million short
children's stories written by GPT-4). Small models learn to write good
English on it, which makes it ideal for a 6 GB GPU.

| Model | Layers × width | Parameters | File |
|-------|----------------|-----------:|------|
| **target** (the one whose text we want) | 8 × 512 | 51M (25M + 26M embedding) | `checkpoints\target.pt` |
| **draft** (the fast guesser) | 2 × 256 | 15M (2M + 13M embedding) | `checkpoints\draft.pt` |
| **distilled draft** (draft that learned from the target) | 2 × 256 | 15M | `checkpoints\draft_distilled.pt` |

Both use the GPT-2 tokenizer (50,257 tokens), so they agree on what a token is
— speculative decoding needs that.

All commands below run in `python\` with `(.venv)` active (Guide 1, section 3).

---

## 1. Prepare the data (10–20 minutes, once)

```bat
python prepare_data.py
```

It downloads TinyStories (~1 GB, only the first time) and writes:

| File | What |
|------|------|
| `data\tinystories\train.bin` | ~470M tokens, stories separated by an end token |
| `data\tinystories\val.bin` | ~4.7M tokens, held out for validation |
| `data\tinystories\meta.json` | tokenizer name, vocabulary size, token counts |

*Quicker:* `python prepare_data.py --max_stories 300000` uses one seventh of
the stories. Fine for a first try; the models will be a bit weaker.

*Your own text:* `python prepare_data.py --source my_text.txt` (stories
separated by blank lines).

---

## 2. Train the target (1–2 hours)

```bat
python train_scratch.py --model target
```

Every 15 seconds it prints a line, and every 250 steps a validation line:

```
step   1250/6000  loss 2.1043  lr 5.71e-04  1.62 steps/s  ETA 49 min
  [eval] step 1250: val_loss 2.0712  val_ppl 7.9334  (best, saved)
```

| Number | Meaning |
|--------|---------|
| `loss` | training loss (cross-entropy). Starts near 10.8, falls fast, then slowly |
| `val_loss` | the same on stories it never trains on. **This is the one that counts** |
| `val_ppl` | perplexity = e^val_loss: "how many tokens it hesitates between". Lower is better |
| `(best, saved)` | best so far, written to `checkpoints\target.pt` |

What to expect at the end (rough):

| Model | val_loss | val_ppl | Writes |
|-------|---------:|--------:|--------|
| random | 10.8 | 50,000 | noise |
| draft | ~2.0–2.5 | ~7–12 | simple sentences, loses the story |
| target | ~1.5–1.9 | ~4.5–7 | coherent little stories |

**Stopping and continuing.** Press **Ctrl+C** any time: it saves
`checkpoints\target_last.pt`. Continue with:

```bat
python train_scratch.py --model target --resume
```

The same works after a crash or a restart. `--resume` on a finished run ends at once.

**Shorter run:** `--max_steps 2000` (about a third of the time, weaker model).

---

## 3. Train the draft (20–40 minutes)

```bat
python train_scratch.py --model draft
```

Same output, same `--resume`. It is 2 layers instead of 8, so it is fast to
train and, more importantly, fast to run.

---

## 4. Distill the draft (20–40 minutes)

```bat
python distill_draft.py --pair scratch
```

The plain draft learned the *data*. Distillation teaches it the *target*: it
starts from `draft.pt` and learns to copy the target's full next-token
probabilities. Every evaluation prints:

```
before distillation: val_loss 0.4120  val_accept 0.6810
  [eval] step 250: val_loss 0.2433  val_accept 0.7614  (best, saved)
```

`val_accept` is the share of draft tokens the target would accept at
temperature 1, computed exactly as `Σ min(p, q)`. **It should go up.** That
number is the whole point of the step.

| Option | Loss | Note |
|--------|------|------|
| `--loss kl` (default) | KL(target ‖ draft) | classic knowledge distillation |
| `--loss tv` | total variation ½·Σ\|p − q\| | acceptance = 1 − TV, so this trains for acceptance directly |
| `--loss kl+tv` | both | |

Each loss writes the same file (`draft_distilled.pt`). To compare them, run one,
benchmark it (Guide 4), rename the file, run the other, benchmark again.

---

## 5. Everything in one go (overnight)

```bat
powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1
```

It runs Guide 2 steps 1–4, the scratch benchmark, then the GPT-2 pair (Guide 3)
and its benchmarks, keeps the PC awake, and logs everything to
`logs\overnight_<date>_<time>.log`. At the end it prints a summary:

```
===== SUMMARY =====
1 prepare data                 OK    (14.2 min)
2 train target                 OK    (71.0 min)
...
```

| Option | Effect |
|--------|--------|
| `-SkipGpt2` | only your own models (no 3.2 GB download) |
| `-WithQwen` | also benchmark the Qwen pair (~4 GB download) |
| `-MaxStories 300000` | smaller dataset, quicker |

If the PC restarts in the night, just run it again: finished steps end at
once, the unfinished one continues from its last checkpoint.

---

## 6. Settings

Everything is in `config.py`, preset `"6gb"`:

| Setting | Default | Change it when |
|---------|---------|----------------|
| `target` `n_layer`, `n_embd` | 8, 512 | bigger GPU → 12, 768 (GPT-2 small size) |
| `micro_batch` | 16 | out of memory → 8 (and double `grad_accum`) |
| `grad_accum` | 4 | keeps the real batch at 16 × 4 × 256 = 16k tokens |
| `max_steps` | 6000 / 4000 | more steps → better model, longer training |
| `block_size` | 256 | longest text the model sees (TinyStories stories fit) |

Logs of every run are saved as CSV in `logs\` (`train_target.csv`,
`train_draft.csv`, `distill_scratch_kl.csv`) if you want to plot the curves.

---

## Problems

| What you see | What to do |
|--------------|-----------|
| `CUDA out of memory` | in `config.py` halve `micro_batch` and double `grad_accum` for that model. Close games / browsers using the GPU |
| `train.bin not found` | run `python prepare_data.py` first |
| very slow (`0.05 steps/s`) | it is running on the CPU. Check `setup.bat`'s GPU line (Guide 1, Problems) |
| loss becomes `nan` | an older GPU on fp16. Lower `lr` in `config.py` by half and `--resume` |
| TinyStories download fails | check the internet; it resumes where it stopped when you run it again |
