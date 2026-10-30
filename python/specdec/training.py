"""One training loop for everything: plain training and draft distillation.

Features: mixed precision (bf16 if the GPU supports it, else fp16 with a
GradScaler), gradient accumulation, warmup + cosine learning rate, gradient
clipping, periodic validation, best-checkpoint saving, resume after a crash
or Ctrl+C, and a CSV log per run.
"""
import csv
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_batch


def pick_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def lr_at(step, lr, warmup, max_steps, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def distill_loss(student_logits, teacher_logits, kind="kl"):
    """How far the draft's next-token distribution is from the target's.

    kl  : KL(p_target || q_draft), the usual knowledge-distillation loss.
    tv  : total variation 0.5*sum|p - q|. Acceptance rate of speculative
          decoding is exactly 1 - TV, so this optimises acceptance directly.
    kl+tv: the sum.
    """
    s = student_logits.float().view(-1, student_logits.size(-1))
    t = teacher_logits.float().view(-1, teacher_logits.size(-1))
    v = min(s.size(-1), t.size(-1))
    s, t = s[:, :v], t[:, :v]
    log_q = F.log_softmax(s, dim=-1)
    log_p = F.log_softmax(t, dim=-1)
    loss = 0.0
    if "kl" in kind:
        loss = loss + F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
    if "tv" in kind:
        loss = loss + 0.5 * (log_p.exp() - log_q.exp()).abs().sum(-1).mean()
    return loss


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def train_loop(*, model, step_loss, evaluate, save, opt, hp, device, run_name, log_dir,
               resume_state=None, log=print):
    """Generic loop.

    step_loss(step) -> loss tensor for ONE micro-batch (already under autocast)
    evaluate()      -> dict of validation numbers, must contain "val_loss"
    save(kind, state) writes a checkpoint; kind is "best" or "last"
    hp: micro_batch, grad_accum, max_steps, lr, warmup, eval_every
    """
    dt = amp_dtype(device)
    ctx = torch.autocast(device_type="cuda", dtype=dt) if dt else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))

    step, best = 0, float("inf")
    if resume_state:
        opt.load_state_dict(resume_state["optimizer"])
        step = resume_state["step"]
        best = resume_state.get("best_val", best)
        log(f"resumed at step {step} (best val loss so far {best:.4f})")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.csv")
    new_file = not (resume_state and os.path.exists(log_path))
    log_f = open(log_path, "w" if new_file else "a", newline="")
    writer = csv.writer(log_f)
    if new_file:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "seconds"])

    t0 = time.time()
    last_print = t0
    running = []
    try:
        while step < hp["max_steps"]:
            lr = lr_at(step, hp["lr"], hp["warmup"], hp["max_steps"])
            for g in opt.param_groups:
                g["lr"] = lr
            model.train()
            total = 0.0
            for _ in range(hp["grad_accum"]):
                with ctx:
                    loss = step_loss(step) / hp["grad_accum"]
                scaler.scale(loss).backward()
                total += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running.append(total)
            step += 1

            now = time.time()
            if now - last_print > 15 or step == 1:
                rate = step_rate(step, t0, now, resume_state)
                eta = (hp["max_steps"] - step) / rate if rate else 0
                log(f"step {step:>6}/{hp['max_steps']}  loss {np.mean(running[-20:]):.4f}  "
                    f"lr {lr:.2e}  {rate:.2f} steps/s  ETA {eta / 60:.0f} min")
                last_print = now

            if step % hp["eval_every"] == 0 or step == hp["max_steps"]:
                with ctx:
                    ev = evaluate()
                improved = ev["val_loss"] < best
                if improved:
                    best = ev["val_loss"]
                extra = "  ".join(f"{k} {v:.4f}" for k, v in ev.items())
                log(f"  [eval] step {step}: {extra}{'  (best, saved)' if improved else ''}")
                writer.writerow([step, f"{np.mean(running[-hp['eval_every']:]):.4f}", f"{ev['val_loss']:.4f}",
                                 f"{lr:.3e}", f"{now - t0:.0f}"])
                log_f.flush()
                state = {"step": step, "optimizer": opt.state_dict(), "best_val": best}
                if improved:
                    save("best", state)
                save("last", state)
    except KeyboardInterrupt:
        log("\nstopped by Ctrl+C -- saving so you can continue with --resume")
        save("last", {"step": step, "optimizer": opt.state_dict(), "best_val": best})
        raise
    finally:
        log_f.close()
    log(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min, best val loss {best:.4f}")
    return best


def step_rate(step, t0, now, resume_state):
    done = step - (resume_state["step"] if resume_state else 0)
    return done / max(1e-9, now - t0)


@torch.no_grad()
def eval_lm_loss(model, val, batches, micro_batch, block_size, device, seed=0):
    """Average cross-entropy on fixed validation windows."""
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        x, y = get_batch(val, micro_batch, block_size, device, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))
