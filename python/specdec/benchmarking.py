"""Measure speculative decoding against the normal baseline and write tables + plots."""
import csv
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from .decoding import autoregressive_generate, speculative_generate
from .pairs import load_lm, exists


def theory_speedup(alpha, gamma, c):
    """Expected speedup for per-token acceptance alpha, draft length gamma, cost ratio c.

    S = (1 - a^(g+1)) / ((1 - a) (g*c + 1))    (Leviathan et al. 2023, Thm 3.8)
    """
    if not np.isfinite(alpha):
        return float("nan")
    if alpha >= 1.0:
        tokens = gamma + 1
    else:
        tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    return tokens / (gamma * c + 1)


@torch.no_grad()
def time_per_token(lm, prompt, reps=30):
    """Average seconds for one single-token forward with a warm cache."""
    cache = lm.new_cache()
    if len(prompt) > 1:
        lm.forward(prompt[:-1], cache)
    base = lm.cache_len(cache)
    sync = (lambda: torch.cuda.synchronize()) if lm.device.type == "cuda" else (lambda: None)
    for _ in range(3):
        lm.forward([prompt[-1]], cache)
        lm.crop(cache, base)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        lm.forward([prompt[-1]], cache)
        lm.crop(cache, base)
    sync()
    return (time.perf_counter() - t0) / reps


def run_benchmark(*, pair_name, pair, tokenizer, prompts, device, out_dir, gammas, temperatures,
                  max_new_tokens, top_k=0, top_p=1.0, seed=1234, prompt_set="", draft_names=None,
                  dtype=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    target = load_lm(pair["target"], device, dtype, "target")
    drafts = {}
    for name, spec in pair["drafts"].items():
        if draft_names and name not in draft_names:
            continue
        if not exists(spec):
            log(f"  skipping draft '{name}': {spec[1]} not found (train / distill it first)")
            continue
        drafts[name] = load_lm(spec, device, dtype, name)
    if not drafts:
        raise SystemExit("no draft model available for this pair")

    eos = getattr(tokenizer, "eot", None)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    log(f"pair '{pair_name}' on {gpu}: target {target.num_params() / 1e6:.1f}M params, drafts: "
        + ", ".join(f"{k} {d.num_params() / 1e6:.1f}M" for k, d in drafts.items()))

    # warm-up (CUDA kernels, allocator) so the first timed run is not penalised
    w = prompts[0]
    autoregressive_generate(target, w, 8)
    for d in drafts.values():
        speculative_generate(target, d, w, 8, gamma=2)

    t_target = time_per_token(target, w)
    cost = {k: time_per_token(d, w) / t_target for k, d in drafts.items()}
    log("  cost ratio c (draft time / target time per token): "
        + ", ".join(f"{k} {v:.3f}" for k, v in cost.items()))

    def gen(seed_offset):
        g = torch.Generator(device=device)
        g.manual_seed(seed + seed_offset)
        return g

    rows, baselines, base_tokens = [], {}, {}
    for temp in temperatures:
        secs, toks = 0.0, 0
        for i, p in enumerate(prompts):
            r = autoregressive_generate(target, p, max_new_tokens, temp, top_k, top_p, eos, gen(i))
            secs += r.stats.seconds
            toks += r.stats.new_tokens
            base_tokens[(temp, i)] = r.tokens
        baselines[temp] = {"seconds": secs, "tokens": toks, "tok_per_s": toks / secs}
        log(f"  baseline T={temp}: {toks / secs:.1f} tokens/s")

        for dname, draft in drafts.items():
            for gamma in gammas:
                secs_s, toks_s, acc, prop, rej, calls, exact = 0.0, 0, 0, 0, 0, 0, 0
                for i, p in enumerate(prompts):
                    r = speculative_generate(target, draft, p, max_new_tokens, gamma, temp, top_k, top_p, eos, gen(i))
                    s = r.stats
                    secs_s += s.seconds
                    toks_s += s.new_tokens
                    acc, prop, rej, calls = acc + s.accepted, prop + s.proposed, rej + s.rejected, calls + s.target_calls
                    exact += int(r.tokens == base_tokens[(temp, i)])
                alpha = acc / (acc + rej) if acc + rej else float("nan")
                row = {
                    "pair": pair_name, "prompts": prompt_set, "draft": dname, "temperature": temp, "gamma": gamma,
                    "accept_rate": acc / prop if prop else float("nan"), "alpha": alpha,
                    "tokens_per_target_call": toks_s / calls, "tok_per_s": toks_s / secs_s,
                    "baseline_tok_per_s": baselines[temp]["tok_per_s"],
                    "speedup": (toks_s / secs_s) / baselines[temp]["tok_per_s"],
                    "theory_speedup": theory_speedup(alpha, gamma, cost[dname]),
                    "cost_ratio": cost[dname],
                    "greedy_exact_match": (exact / len(prompts)) if temp == 0 else None,
                }
                rows.append(row)
                em = f"  exact {row['greedy_exact_match']:.0%}" if temp == 0 else ""
                log(f"  {dname:>9} T={temp} gamma={gamma}: accept {row['accept_rate']:.2f}  "
                    f"tok/call {row['tokens_per_target_call']:.2f}  {row['tok_per_s']:.1f} tok/s  "
                    f"speedup {row['speedup']:.2f}x (theory {row['theory_speedup']:.2f}x){em}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = os.path.join(out_dir, f"{pair_name}_{prompt_set}_{stamp}")
    meta = {"pair": pair_name, "prompts": prompt_set, "date": stamp, "device": gpu,
            "target_params_M": target.num_params() / 1e6,
            "draft_params_M": {k: d.num_params() / 1e6 for k, d in drafts.items()},
            "cost_ratio": cost, "baselines": {str(k): v for k, v in baselines.items()},
            "n_prompts": len(prompts), "max_new_tokens": max_new_tokens, "top_k": top_k, "top_p": top_p}
    with open(base + ".json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, base + ".png", f"{pair_name} ({prompt_set}) on {gpu}")
    write_summary(out_dir)
    log(f"  saved {base}.json / .csv / .png and {os.path.join(out_dir, 'RESULTS.md')}")
    return rows, meta


def plot(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    keys = sorted({(r["draft"], r["temperature"]) for r in rows})
    for (d, t) in keys:
        rs = [r for r in rows if r["draft"] == d and r["temperature"] == t]
        g = [r["gamma"] for r in rs]
        line, = ax1.plot(g, [r["speedup"] for r in rs], "o-", label=f"{d}, T={t}")
        ax1.plot(g, [r["theory_speedup"] for r in rs], "--", color=line.get_color(), alpha=0.5)
        ax2.plot(g, [r["accept_rate"] for r in rs], "o-", color=line.get_color(), label=f"{d}, T={t}")
    ax1.axhline(1.0, color="grey", lw=1)
    ax1.set(xlabel="gamma (draft tokens per round)", ylabel="speedup vs normal decoding",
            title="Speedup (solid = measured, dashed = theory)")
    ax2.set(xlabel="gamma", ylabel="share of draft tokens accepted", title="Acceptance", ylim=(0, 1))
    from matplotlib.ticker import MaxNLocator
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(out_dir):
    """RESULTS.md: for the newest run of each pair/prompt set, the best gamma per draft and temperature."""
    latest = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        if "meta" not in data:
            continue
        latest[(data["meta"]["pair"], data["meta"]["prompts"])] = (path, data)
    lines = ["# Results", "", "Written by `benchmark.py`. Newest run per pair and prompt set. "
             "Speedup = tokens/s with speculative decoding / tokens/s with normal decoding "
             "(same target model, same prompts).", ""]
    for (pair, prompts), (path, data) in sorted(latest.items()):
        m = data["meta"]
        lines += [f"## {pair} — {prompts} prompts ({m['date']}, {m['device']})", "",
                  f"Target {m['target_params_M']:.1f}M params. Drafts: "
                  + ", ".join(f"{k} {v:.1f}M (c = {m['cost_ratio'][k]:.3f})" for k, v in m["draft_params_M"].items())
                  + f". {m['n_prompts']} prompts x {m['max_new_tokens']} new tokens.", "",
                  "| Draft | Temp | Best γ | Accepted | Tokens / target call | Tokens/s (normal → spec) | Speedup | Theory | Greedy exact |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        groups = sorted({(r["draft"], r["temperature"]) for r in data["rows"]})
        for d, t in groups:
            best = max((r for r in data["rows"] if r["draft"] == d and r["temperature"] == t), key=lambda r: r["speedup"])
            ex = "—" if best["greedy_exact_match"] is None else f"{best['greedy_exact_match']:.0%}"
            lines.append(f"| {d} | {t} | {best['gamma']} | {best['accept_rate']:.0%} | "
                         f"{best['tokens_per_target_call']:.2f} | {best['baseline_tok_per_s']:.0f} → {best['tok_per_s']:.0f} | "
                         f"**{best['speedup']:.2f}×** | {best['theory_speedup']:.2f}× | {ex} |")
        lines += ["", f"![plot]({os.path.basename(path)[:-5]}.png)", ""]
    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
