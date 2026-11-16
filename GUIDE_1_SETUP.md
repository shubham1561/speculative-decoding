# Guide 1 — Setup

## 1. What you need

| Thing | Why |
|-------|-----|
| Windows 10/11 | the launchers are `.bat` and `.ps1` (the Python code itself runs anywhere) |
| Python 3.10 – 3.13 from python.org | tick **"Add python.exe to PATH"** when installing |
| An NVIDIA GPU, 6 GB is enough | every default setting fits 6 GB |
| An up-to-date NVIDIA driver | PyTorch brings its own CUDA; the driver is all you need |
| ~15 GB free disk | PyTorch 2.5 GB, TinyStories 2 GB, GPT-2 large 3.2 GB, checkpoints |

---

## 2. Install (once)

Open a terminal in the repo's `python\` folder (Start → `cmd` → Enter, then `cd` there) and run:

```bat
setup.bat
```

It makes `python\.venv`, installs PyTorch **with GPU support**, then the rest.
At the end it prints something like:

```
PyTorch 2.x.x+cu126
GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6.4 GB
```

If it says **NOT FOUND**, see *Problems* below before training anything.

---

## 3. Open the terminal (every time)

```bat
cd <repo>\python
.venv\Scripts\activate
```

You should see `(.venv)` at the start of the line. Every command in the guides assumes this.

---

## 4. Check the machinery (about 1 minute)

```bat
smoke_test.bat
```

or, with the environment active:

```bat
python smoke_test.py
python -m pytest
```

`smoke_test.py` makes up a few thousand tiny stories and runs **every step**
of the real pipeline with tiny models: prepare data → train target → train
draft → distill → benchmark. It works in a temporary folder and does **not**
touch your real data, checkpoints or results. You want:

```
  [PASS] data files written
  [PASS] target trains (val loss 1.6, random would be 5.55)
  [PASS] draft trains (val loss 2.4)
  [PASS] distillation runs and saves
  [PASS] greedy speculative == greedy normal (100% of prompts identical)
  [PASS] draft tokens get accepted
  [PASS] results table and plot written
ALL CHECKS PASSED
```

`pytest` should end with `19 passed`. The test that takes longest samples
4,000 outputs to check the probabilities — about 20 seconds.

The tiny models write nonsense. That is expected: the smoke test checks that
the code works, not that the models are good.

---

## Problems

| What you see | What to do |
|--------------|-----------|
| `'python' is not recognized` | Python is not on PATH. Reinstall it and tick "Add python.exe to PATH", or use `py -3.12 -m venv .venv` in place of `python -m venv .venv` |
| `GPU: NOT FOUND` | update the NVIDIA driver, restart, run `setup.bat` again. If you installed torch yourself with `pip install torch`, that is the CPU build: `.venv\Scripts\python -m pip uninstall torch` then run `setup.bat` |
| torch install says *no matching distribution* | your Python is too new for the CUDA wheels. Install Python 3.12 and delete `.venv`, then `setup.bat` |
| an older GPU (GTX 10xx) | works; it uses fp16 instead of bf16 automatically. In `setup.bat` you can swap `cu126` for `cu118` |
| `running scripts is disabled on this system` | start PowerShell scripts as `powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1` |
