"""All settings in one place. Change numbers here, not inside the scripts.

The defaults are sized for a 6 GB GPU (preset "6gb"). The "tiny" preset is
what smoke_test.py uses on the CPU.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(HERE, "data", "tinystories")     # train.bin / val.bin / meta.json
CKPT_DIR = os.path.join(HERE, "checkpoints")             # trained models
RESULTS_DIR = os.path.join(HERE, "results")              # benchmark tables and plots
LOG_DIR = os.path.join(HERE, "logs")                     # training logs (csv)

# ---------------------------------------------------------------------------
# Models you train yourself (from scratch, on TinyStories, GPT-2 tokenizer)
# ---------------------------------------------------------------------------
PRESETS = {
    "6gb": {
        # ~51M parameters (25M + 26M tied embedding). The model whose output we want.
        "target": dict(n_layer=8, n_head=8, n_embd=512, block_size=256, dropout=0.0),
        # ~15M parameters (2M + 13M embedding). Fast guesser: 2 layers instead of 8.
        "draft": dict(n_layer=2, n_head=4, n_embd=256, block_size=256, dropout=0.0),
        "train": {
            "target": dict(micro_batch=16, grad_accum=4, max_steps=6000, lr=6e-4, warmup=300,
                           eval_every=250, eval_batches=40),
            "draft": dict(micro_batch=16, grad_accum=4, max_steps=4000, lr=1e-3, warmup=200,
                          eval_every=250, eval_batches=40),
        },
        # knowledge distillation of the draft from YOUR target (raises acceptance)
        "distill_scratch": dict(micro_batch=8, grad_accum=8, max_steps=1500, lr=3e-4, warmup=100,
                                eval_every=250, eval_batches=20),
        # knowledge distillation of distilgpt2 from gpt2-large
        "distill_hf": dict(micro_batch=4, grad_accum=8, max_steps=1000, lr=5e-5, warmup=50,
                           eval_every=250, eval_batches=10, block_size=256),
    },
    "tiny": {
        "target": dict(n_layer=2, n_head=2, n_embd=64, block_size=128, dropout=0.0),
        "draft": dict(n_layer=1, n_head=2, n_embd=32, block_size=128, dropout=0.0),
        "train": {
            "target": dict(micro_batch=8, grad_accum=1, max_steps=150, lr=3e-3, warmup=10,
                           eval_every=50, eval_batches=4),
            "draft": dict(micro_batch=8, grad_accum=1, max_steps=150, lr=3e-3, warmup=10,
                          eval_every=50, eval_batches=4),
        },
        "distill_scratch": dict(micro_batch=8, grad_accum=1, max_steps=60, lr=1e-3, warmup=5,
                                eval_every=30, eval_batches=4),
        "distill_hf": dict(micro_batch=2, grad_accum=1, max_steps=10, lr=1e-4, warmup=2,
                           eval_every=5, eval_batches=2, block_size=64),
    },
}

# ---------------------------------------------------------------------------
# Target / draft pairs the benchmark knows about
#   own:  a checkpoint file in checkpoints/
#   hf:   a HuggingFace model id, or a folder in checkpoints/
# A draft that does not exist yet is skipped (with a message).
# ---------------------------------------------------------------------------
PAIRS = {
    "scratch": {
        "tokenizer": "gpt2",
        "target": ("own", os.path.join(CKPT_DIR, "target.pt")),
        "drafts": {
            "plain": ("own", os.path.join(CKPT_DIR, "draft.pt")),
            "distilled": ("own", os.path.join(CKPT_DIR, "draft_distilled.pt")),
        },
    },
    "gpt2": {                                    # 774M target + 82M draft, fp16 ~1.8 GB total
        "tokenizer": "gpt2",
        "target": ("hf", "gpt2-large"),
        "drafts": {
            "plain": ("hf", "distilgpt2"),
            "distilled": ("hf", os.path.join(CKPT_DIR, "distilgpt2_distilled")),
        },
    },
    "qwen": {                                    # 1.5B target + 0.5B draft, ~4 GB total (inference only)
        "tokenizer": "Qwen/Qwen2.5-1.5B",
        "dtype": "bfloat16",
        "target": ("hf", "Qwen/Qwen2.5-1.5B"),
        "drafts": {
            "plain": ("hf", "Qwen/Qwen2.5-0.5B"),
        },
    },
}

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------
BENCH = dict(
    gammas=[1, 2, 3, 4, 5, 6, 8],
    temperatures=[0.0, 0.8],
    n_prompts=20,
    prompt_tokens=32,
    max_new_tokens=128,
    top_k=0,
    top_p=1.0,
    seed=1234,
)

# Prompts used when --prompts general (any topic, not stories)
GENERAL_PROMPTS = [
    "The history of the printing press begins",
    "In computer science, a hash table is",
    "The main causes of the French Revolution were",
    "To make a good cup of coffee, you should",
    "Photosynthesis is the process by which",
    "The stock market crashed in 1929 because",
    "A neural network learns by",
    "The city of Bengaluru is known for",
    "Climate change affects agriculture by",
    "def fibonacci(n):\n    ",
    "The theory of relativity states that",
    "The best way to learn a new language is",
    "Once upon a time, in a small village,",
    "The Great Wall of China was built",
    "Electric cars are becoming popular because",
    "The human heart has four chambers:",
    "In the year 2050, cities will",
    "Machine learning differs from traditional programming in that",
    "The recipe calls for two cups of flour,",
    "The football match ended in a draw after",
]
