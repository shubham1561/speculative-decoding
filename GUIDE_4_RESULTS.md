# Guide 4 — Measure, read the results, put them on GitHub

## 1. Try it: type a few words, the model completes them

**Easiest — the browser demo.** Double-click **`START_DEMO.bat`** in the repo folder.
A black window opens (the local server — keep it open), the models load (10–30 s),
and your browser opens `http://127.0.0.1:8000`. Type a prompt and press Enter:

- **left:** normal decoding, one big-model pass per token
- **right:** speculative decoding — green = draft guess accepted, yellow = written by the big model
- both runs are replayed at the speed they were measured, so you see which finishes first
- below: speedup, big-model passes (e.g. *42 vs 83*), share of draft guesses accepted, and
  whether the text is identical

Change the model pair, draft, γ, temperature and length on the page. The GPU is warmed up
before each timed run, so the comparison is fair. Close the black window to stop the demo.
The page only works on your own computer (the server listens on 127.0.0.1).

**In the terminal** (same thing, text only):

```bat
python generate.py --interactive
```

It loads the models once, then waits for you. Type a few words, press Enter,
and it finishes the sentence (example — your words and numbers will differ):

```
you> Lily found a red ball
model> Lily found a red ball in the park. 
speculative 410 tok/s, 9/12 draft tokens accepted (75%), 2.8 tokens per target call | normal 190 tok/s | speedup 2.16x | same text: YES
```

| Colour | Meaning |
|--------|---------|
| **green** | guessed by the draft and accepted by the target — came for free |
| **yellow** | made by the target itself (a correction after a rejection, or the bonus token) |

**The text is the same as normal decoding** — `same text: YES`. Speculative
decoding does not write different or longer text; it writes the *same* text
with fewer passes of the big model. The green words are what you saved.

Commands while it runs:

| Type | Effect |
|------|--------|
| `/until sentence` | stop at the end of the sentence (default) |
| `/until line` | stop at the end of the line / paragraph |
| `/until none` | keep going up to `/tokens` |
| `/tokens 100` | most new tokens per completion (default 60) |
| `/gamma 5` | draft tokens per round — try 1 to 8 and watch the speedup |
| `/temp 0.8` | more varied completions (0 = always the most likely word) |
| `/compare` | switch the normal-decoding comparison off/on (off = faster demo) |
| `/quit` | exit; prints the overall speedup of the session |

Your own models learned from children's stories, so they turn everything into a
story. For any topic use the GPT-2 pair (Guide 3):

```bat
python generate.py --interactive --pair gpt2
```

**One prompt, no typing** (prints the full report):

```bat
python generate.py --prompt "Once upon a time, a little dog"
python generate.py --pair gpt2 --prompt "The history of Rome" --gamma 5
python generate.py --draft plain       :: compare with the distilled draft
```

Short completions (one sentence, ~10–20 tokens) give noisy speed numbers.
For the real measurement, use the benchmark below.

---

## 2. Run the benchmark

```bat
python benchmark.py --pair scratch
python benchmark.py --pair gpt2 --prompts stories
python benchmark.py --pair gpt2 --prompts general
```

For every draft × temperature (0 and 0.8) × γ (1, 2, 3, 4, 5, 6, 8) it runs the
same 20 prompts with normal and with speculative decoding, 128 new tokens each.
Quick version: `--gammas 2 4 --temperatures 0 --n_prompts 5`.

It writes to `results\`:

| File | What |
|------|------|
| `<pair>_<prompts>_<date>.json` / `.csv` | every number |
| `<pair>_<prompts>_<date>.png` | speedup and acceptance against γ |
| `RESULTS.md` | the summary table: best γ for each draft and temperature |

---

## 3. Read the results — in this order

**① Greedy exact.** At temperature 0, speculative output must be the same text
as normal decoding. Expect **100%**. (On the GPU, bf16/fp16 rounding can very
rarely flip a near-tie between two tokens, because the target sees 1 token at a
time in one case and γ+1 in the other. One prompt out of 20 is that; many is a bug.)

**② Accepted.** The share of draft tokens the target kept.

| Accepted | Meaning |
|----------|---------|
| > 70% | very good draft |
| 50–70% | useful |
| < 40% | the draft disagrees too often — distil it (Guide 2, step 4) |

**③ Tokens per target call.** How many tokens one big-model pass produces.
Normal decoding = 1.00. This is the gain *before* paying for the draft.

**④ Speedup.** tokens/s speculative ÷ tokens/s normal, same prompts. Above
1.0 = faster. This is the headline number.

**⑤ Theory.** The speedup the formula predicts from the measured acceptance α
and cost ratio c (draft time ÷ target time per token):

```
S(γ) = (1 − α^(γ+1)) / ((1 − α)(γ·c + 1))
```

Measured is usually a bit **below** theory: the formula ignores Python
overhead, the verify pass costing a little more than one token, and the
draft-side bookkeeping. A big gap means overhead dominates — typical for very
small models, where every forward pass is tiny.

**What to expect** (rough, depends on your GPU):

| Pair | c | Likely best γ | Likely speedup |
|------|--:|--------------:|---------------:|
| scratch (51M / 15M) | ~0.3 | 2–4 | 1.1–1.6× |
| gpt2 (774M / 82M) | ~0.15–0.25 | 3–5 | 1.3–2× |
| qwen (1.5B / 0.5B) | ~0.5 | 2–3 | 1.0–1.4× |

Speedup is highest when the target is **much** bigger than the draft (small c)
and they agree often (high α). Higher temperature usually lowers α.

---

## 4. Put the results in the README

1. Open `python\results\RESULTS.md`.
2. Copy the numbers into the **Results** table of the main `README.md`
   (best γ, accepted, speedup, greedy identical).
3. Add the plot: `![speedup](python/results/<file>.png)`.
4. Write 2–3 sentences: what helped most (distillation? domain?), and where
   measured and theory differ.

`results\` **is** uploaded to GitHub (small files). `data\`, `checkpoints\`,
`logs\` and `.venv\` are **not** (see `.gitignore`) — they are huge and anyone
can recreate them with the scripts.

---

## 5. Upload to GitHub

The folder is already a git repository with a first commit. On github.com,
make a **new empty repository** (no README, no licence — they exist already),
for example `speculative-decoding`. Then, in the repo folder:

```bat
git log --oneline                       :: shows the first commit
git remote add origin https://github.com/<your-username>/speculative-decoding.git
git branch -M main
git push -u origin main
```

After you add results:

```bat
git add README.md python/results
git commit -m "Add benchmark results"
git push
```

*Want a different name or e-mail on the first commit?*
`git config user.name "..."`, `git config user.email "..."`, then
`git commit --amend --reset-author --no-edit` before the first push.

---

## Problems

| What you see | What to do |
|--------------|-----------|
| speedup below 1.0 for every γ | normal for tiny drafts on a fast GPU when acceptance is low. Use the distilled draft, try γ = 2, check c in the log (c above 0.5 rarely pays off) |
| greedy exact well below 100% | run `python -m pytest`; if the tests pass, it is GPU rounding — try `--device cpu` on a few prompts to confirm 100% there |
| `no draft model available` | train or distil the draft first (Guide 2 / Guide 3) |
| the colours show as `[92m` | use Windows Terminal, or ignore them — the numbers below are what matter |
| the first interactive line takes a few seconds | it is finding which tokens end a sentence, once per `/until` mode |
