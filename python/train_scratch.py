"""Step 2: train your own target and draft models from scratch.

    python train_scratch.py --model target        # ~51M params -> checkpoints/target.pt
    python train_scratch.py --model draft         # ~15M params -> checkpoints/draft.pt
    python train_scratch.py --model target --resume   # continue after a stop / Ctrl+C
    python train_scratch.py --model target --max_steps 1000   # quick, weaker run

Model sizes and training settings come from config.PRESETS["6gb"].
"""
import argparse
import os

import numpy as np
import torch

import config
from specdec.data import load_split, read_meta, get_batch
from specdec.gpt import GPT, GPTConfig
from specdec.training import pick_device, train_loop, eval_lm_loss


def run(model_name="target", preset="6gb", data_dir=config.DATA_DIR, ckpt_dir=config.CKPT_DIR,
        log_dir=config.LOG_DIR, device="auto", resume=False, max_steps=None, seed=0, out_name=None, log=print):
    dev = pick_device(device)
    torch.manual_seed(seed)
    meta = read_meta(data_dir)
    mcfg = GPTConfig(vocab_size=meta["vocab_size"], **config.PRESETS[preset][model_name])
    hp = dict(config.PRESETS[preset]["train"][model_name])
    if max_steps:
        hp["max_steps"] = max_steps
    out_name = out_name or model_name
    best_path = os.path.join(ckpt_dir, f"{out_name}.pt")
    last_path = os.path.join(ckpt_dir, f"{out_name}_last.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = GPT(mcfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], betas=(0.9, 0.95), weight_decay=0.1,
                            fused=(dev.type == "cuda"))
    resume_state = None
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        resume_state = ck
    elif resume:
        log(f"no {last_path} yet -- starting fresh")

    train, val = load_split(data_dir, "train"), load_split(data_dir, "val")
    rng = np.random.default_rng(seed + (resume_state["step"] if resume_state else 0))
    log(f"training {out_name}: {model.num_params() / 1e6:.1f}M params (+{mcfg.vocab_size * mcfg.n_embd / 1e6:.1f}M "
        f"embedding), {len(train) / 1e6:.0f}M training tokens, device {dev}")
    log(f"  {hp['max_steps']} steps x {hp['micro_batch']} x {hp['grad_accum']} x {mcfg.block_size} tokens "
        f"= {hp['max_steps'] * hp['micro_batch'] * hp['grad_accum'] * mcfg.block_size / 1e6:.0f}M tokens seen")

    def step_loss(step):
        x, y = get_batch(train, hp["micro_batch"], mcfg.block_size, dev, rng)
        return model(x, y)[1]

    def evaluate():
        vl = eval_lm_loss(model, val, hp["eval_batches"], hp["micro_batch"], mcfg.block_size, dev)
        return {"val_loss": vl, "val_ppl": float(np.exp(vl))}

    def save(kind, state):
        ck = {"model": model.state_dict(), "config": mcfg.to_dict(), "tokenizer": meta["tokenizer"],
              "preset": preset, **state}
        if kind == "best":
            ck = {k: v for k, v in ck.items() if k != "optimizer"}    # small file for inference
        torch.save(ck, best_path if kind == "best" else last_path)

    best = train_loop(model=model, step_loss=step_loss, evaluate=evaluate, save=save, opt=opt, hp=hp,
                      device=dev, run_name=f"train_{out_name}", log_dir=log_dir, resume_state=resume_state, log=log)
    log(f"saved best model to {best_path}")
    return best_path, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["target", "draft"], required=True)
    ap.add_argument("--preset", default="6gb", choices=list(config.PRESETS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.model, a.preset, device=a.device, resume=a.resume, max_steps=a.max_steps, seed=a.seed)


if __name__ == "__main__":
    main()
