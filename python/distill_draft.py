"""Step 3: make the draft guess more like the target (knowledge distillation).

The draft is trained to copy the target's next-token PROBABILITIES, not just
the next word. A draft that thinks like the target gets more tokens accepted.

    python distill_draft.py --pair scratch          # your draft.pt  <- your target.pt
    python distill_draft.py --pair gpt2             # distilgpt2     <- gpt2-large
    python distill_draft.py --pair scratch --loss tv   # optimise acceptance (1 - TV) directly

Output:
    scratch -> checkpoints/draft_distilled.pt
    gpt2    -> checkpoints/distilgpt2_distilled/   (a HuggingFace folder)

Every evaluation prints val_accept: the expected share of draft tokens the target
would accept at temperature 1 (= mean of sum_x min(p, q)). Watch it go up.
"""
import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch

import config
from specdec.data import load_split, read_meta, get_batch
from specdec.gpt import GPT, GPTConfig, load_gpt
from specdec.pairs import load_hf_model
from specdec.training import pick_device, amp_dtype, train_loop, distill_loss


def _logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out.logits


@torch.no_grad()
def eval_pair(teacher, student, val, hp, block, dev, kind):
    student.eval()
    rng = np.random.default_rng(0)
    losses, accepts = [], []
    for _ in range(hp["eval_batches"]):
        x, _ = get_batch(val, hp["micro_batch"], block, dev, rng)
        t, s = _logits(teacher, x).float(), _logits(student, x).float()
        v = min(t.size(-1), s.size(-1))
        t, s = t[..., :v], s[..., :v]
        losses.append(distill_loss(s, t, kind).item())
        accepts.append(torch.minimum(t.softmax(-1), s.softmax(-1)).sum(-1).mean().item())
    return {"val_loss": float(np.mean(losses)), "val_accept": float(np.mean(accepts))}


def run(pair="scratch", preset="6gb", loss="kl", data_dir=config.DATA_DIR, ckpt_dir=config.CKPT_DIR,
        log_dir=config.LOG_DIR, device="auto", resume=False, max_steps=None, seed=0,
        teacher_path=None, student_path=None, out_path=None, log=print):
    dev = pick_device(device)
    torch.manual_seed(seed)
    meta = read_meta(data_dir)
    teacher_dtype = amp_dtype(dev) or torch.float32

    if pair == "scratch":
        hp = dict(config.PRESETS[preset]["distill_scratch"])
        teacher_path = teacher_path or os.path.join(ckpt_dir, "target.pt")
        student_path = student_path or os.path.join(ckpt_dir, "draft.pt")
        out_path = out_path or os.path.join(ckpt_dir, "draft_distilled.pt")
        teacher = load_gpt(teacher_path, dev, teacher_dtype)
        ck = torch.load(student_path, map_location="cpu", weights_only=False)
        scfg = GPTConfig(**ck["config"])
        student = GPT(scfg).to(dev)
        student.load_state_dict(ck["model"])
        block = min(scfg.block_size, teacher.cfg.block_size)
    elif pair == "gpt2":
        if meta["tokenizer"] != "gpt2":
            raise SystemExit("the gpt2 pair needs data prepared with --tokenizer gpt2")
        hp = dict(config.PRESETS[preset]["distill_hf"])
        teacher_name = teacher_path or config.PAIRS["gpt2"]["target"][1]
        student_name = student_path or config.PAIRS["gpt2"]["drafts"]["plain"][1]
        out_path = out_path or config.PAIRS["gpt2"]["drafts"]["distilled"][1]
        log(f"loading teacher {teacher_name} and student {student_name} (downloaded the first time)")
        teacher = load_hf_model(teacher_name, dev, teacher_dtype)
        student = load_hf_model(student_name, dev, torch.float32)
        block = hp["block_size"]
    else:
        raise SystemExit(f"unknown pair {pair}")

    if max_steps:
        hp["max_steps"] = max_steps
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    last_path = (out_path[:-3] if out_path.endswith(".pt") else out_path) + "_last.pt"
    opt = torch.optim.AdamW(student.parameters(), lr=hp["lr"], betas=(0.9, 0.95), weight_decay=0.01)
    resume_state = None
    if resume and os.path.exists(last_path):
        resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
        student.load_state_dict(resume_state["model"])

    train, val = load_split(data_dir, "train"), load_split(data_dir, "val")
    rng = np.random.default_rng(seed + 1 + (resume_state["step"] if resume_state else 0))

    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype(dev)) if amp_dtype(dev) else nullcontext()
    with torch.no_grad(), ctx:
        before = eval_pair(teacher, student, val, hp, block, dev, loss)
    log(f"before distillation: val_loss {before['val_loss']:.4f}  val_accept {before['val_accept']:.4f}")

    def step_loss(step):
        x, _ = get_batch(train, hp["micro_batch"], block, dev, rng)
        with torch.no_grad():
            t = _logits(teacher, x)
        return distill_loss(_logits(student, x), t, loss)

    def evaluate():
        return eval_pair(teacher, student, val, hp, block, dev, loss)

    def save(kind, state):
        os.makedirs(os.path.dirname(last_path), exist_ok=True)
        if kind == "last":
            torch.save({"model": student.state_dict(), **state}, last_path)
            return
        if pair == "scratch":
            torch.save({"model": student.state_dict(), "config": scfg.to_dict(), "tokenizer": meta["tokenizer"],
                        "distilled_from": os.path.basename(teacher_path), "loss": loss,
                        "step": state["step"], "best_val": state["best_val"]}, out_path)
        else:
            student.save_pretrained(out_path)

    best = train_loop(model=student, step_loss=step_loss, evaluate=evaluate, save=save, opt=opt, hp=hp,
                      device=dev, run_name=f"distill_{pair}_{loss}", log_dir=log_dir, resume_state=resume_state,
                      log=log)
    log(f"saved distilled draft to {out_path}")
    return out_path, before, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=["scratch", "gpt2"], required=True)
    ap.add_argument("--loss", choices=["kl", "tv", "kl+tv"], default="kl")
    ap.add_argument("--preset", default="6gb", choices=list(config.PRESETS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    run(a.pair, a.preset, a.loss, device=a.device, resume=a.resume, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
