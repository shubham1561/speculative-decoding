"""See speculative decoding happen: type a few words, get them completed.

One prompt:
    python generate.py --prompt "Once upon a time, a little dog"
    python generate.py --pair gpt2 --prompt "The history of Rome" --gamma 5 --temperature 0.8
    python generate.py --draft plain        # compare with --draft distilled

Interactive (keeps the models loaded, completes every line you type):
    python generate.py --interactive
    python generate.py --interactive --pair gpt2 --until line

    In interactive mode, type a few words and press Enter. Commands:
        /gamma 5     /temp 0.8     /tokens 100     /until sentence|line|none
        /compare     (normal decoding too: on/off)  /help     /quit

Colours in the output:
    green  = guessed by the draft and accepted by the target (came "for free")
    yellow = produced by the target itself (a correction, or the bonus token)
Then it runs normal decoding on the same prompt and prints both speeds.
The text is the same either way -- speculative decoding only changes the speed.
"""
import argparse
import os

import torch

import config
from specdec.data import get_tokenizer
from specdec.decoding import autoregressive_generate, speculative_generate
from specdec.pairs import load_lm, exists, inference_dtype
from specdec.training import pick_device

GREEN, YELLOW, DIM, BOLD, RESET = "\033[92m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
SENTENCE_END = (".", "!", "?")
CLOSERS = " \"'”’)]"


def stop_token_ids(tok, until):
    """Token ids that end a completion: any token with a line break, or (sentence) ending in . ! ?"""
    if until == "none":
        return set()
    ids = set()
    for i in range(tok.vocab_size):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if "\n" in s or (until == "sentence" and s.rstrip(CLOSERS).endswith(SENTENCE_END)):
            ids.add(i)
    return ids


class Completer:
    """Loads the pair once; complete() runs speculative (and optionally normal) decoding."""

    def __init__(self, a):
        self.a = a
        self.dev = pick_device(a.device)
        pair = config.PAIRS[a.pair]
        names = [a.draft] if a.draft else ["distilled", "plain"]
        self.dname = next((n for n in names if n in pair["drafts"] and exists(pair["drafts"][n])), None)
        if self.dname is None:
            raise SystemExit(f"no draft found for pair '{a.pair}' -- train it first (see the guides)")
        self.tok = get_tokenizer(pair["tokenizer"])
        dtype = inference_dtype(self.dev, pair.get("dtype"))
        print(f"loading pair '{a.pair}' (draft '{self.dname}') on {self.dev} ...", flush=True)
        self.target = load_lm(pair["target"], self.dev, dtype)
        self.draft = load_lm(pair["drafts"][self.dname], self.dev, dtype)
        self.eos = getattr(self.tok, "eot", None)
        self._stops = {}
        self.totals = {"spec_tokens": 0, "spec_s": 0.0, "base_tokens": 0, "base_s": 0.0, "lines": 0}
        warm = self.tok.encode("Once upon a time") or [0]
        speculative_generate(self.target, self.draft, warm, 8, 2)       # warm-up (CUDA kernels)
        autoregressive_generate(self.target, warm, 8)

    def stops(self, until):
        if until not in self._stops:
            self._stops[until] = stop_token_ids(self.tok, until)
        return self._stops[until]

    def complete(self, text, gamma, temperature, max_new_tokens, until, compare, seed):
        a = self.a
        prompt = self.tok.encode(text)
        if not prompt:
            return None
        room = min(m for m in (self.target.max_len, self.draft.max_len) if m) - max_new_tokens
        prompt = prompt[-max(1, room):]                       # keep the end of a very long prompt
        kw = dict(temperature=temperature, top_k=a.top_k, top_p=a.top_p, eos_id=self.eos,
                  stop_ids=self.stops(until))
        g = torch.Generator(device=self.dev).manual_seed(seed)
        spec = speculative_generate(self.target, self.draft, prompt, max_new_tokens, gamma, generator=g, **kw)
        base = None
        if compare:
            g = torch.Generator(device=self.dev).manual_seed(seed)
            base = autoregressive_generate(self.target, prompt, max_new_tokens, generator=g, **kw)
        t = self.totals
        t["spec_tokens"] += spec.stats.new_tokens
        t["spec_s"] += spec.stats.seconds
        t["lines"] += 1
        if base:
            t["base_tokens"] += base.stats.new_tokens
            t["base_s"] += base.stats.seconds
        return spec, base

    def coloured(self, tokens, sources):
        return "".join((GREEN if src == "draft" else YELLOW) + self.tok.decode([t]) + RESET
                       for t, src in zip(tokens, sources))

    def report(self, spec, base, temperature, short=False):
        s = spec.stats
        acc = f"{s.accepted}/{s.proposed} draft tokens accepted ({s.acceptance_rate:.0%})" if s.proposed else "no draft tokens"
        if short:
            line = f"{DIM}speculative {s.tokens_per_second:.0f} tok/s, {acc}, {s.tokens_per_target_call:.2f} tokens per target call"
            if base:
                b = base.stats
                line += f" | normal {b.tokens_per_second:.0f} tok/s | speedup {s.tokens_per_second / b.tokens_per_second:.2f}x"
                if temperature == 0:
                    line += f" | same text: {'YES' if spec.tokens == base.tokens else 'NO'}"
            print(line + RESET)
            return
        print(f"\n{GREEN}green{RESET} = draft token accepted, {YELLOW}yellow{RESET} = made by the target\n")
        print(f"speculative: {s.new_tokens} tokens in {s.seconds:.2f} s = {s.tokens_per_second:.1f} tok/s, "
              f"{acc}, {s.tokens_per_target_call:.2f} tokens per target call")
        if base:
            b = base.stats
            print(f"normal:      {b.new_tokens} tokens in {b.seconds:.2f} s = {b.tokens_per_second:.1f} tok/s")
            print(f"speedup:     {s.tokens_per_second / b.tokens_per_second:.2f}x")
            if temperature == 0:
                print(f"same text as normal greedy decoding: "
                      f"{'YES' if spec.tokens == base.tokens else 'NO (see GUIDE_4, Problems)'}")


HELP = f"""{BOLD}Type a few words and press Enter{RESET} -- the model completes them.
  /gamma N          draft tokens per round (now {{gamma}})
  /temp X           0 = always the most likely word; 0.7-1.0 = more varied (now {{temp}})
  /tokens N         most new tokens per completion (now {{tokens}})
  /until MODE       sentence | line | none -- where a completion stops (now {{until}})
  /compare          also run normal decoding and show the speedup: on/off (now {{compare}})
  /help             this list
  /quit             exit (or Ctrl+C)"""


def interactive(c: Completer, a):
    st = dict(gamma=a.gamma, temp=a.temperature, tokens=a.max_new_tokens, until=a.until, compare=True)
    seed = a.seed
    print(f"\n{BOLD}Speculative decoding -- interactive{RESET}  (pair {a.pair}, draft '{c.dname}')")
    if a.pair == "scratch":
        print(f"{DIM}Your own models learned from children's stories, so they write stories. "
              f"For any topic use --pair gpt2.{RESET}")
    print(HELP.format(**{**st, "compare": "on" if st["compare"] else "off"}))
    print(f"{GREEN}green{RESET} = draft guess accepted (free), {YELLOW}yellow{RESET} = written by the target\n")
    while True:
        try:
            text = input(f"{BOLD}you>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        if text.startswith("/"):
            cmd, _, arg = text[1:].strip().partition(" ")
            arg = arg.strip()
            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "gamma":
                    st["gamma"] = max(0, int(arg))
                elif cmd in ("temp", "temperature"):
                    st["temp"] = max(0.0, float(arg))
                elif cmd == "tokens":
                    st["tokens"] = max(1, int(arg))
                elif cmd == "until":
                    if arg not in ("sentence", "line", "none"):
                        raise ValueError
                    st["until"] = arg
                elif cmd == "compare":
                    st["compare"] = not st["compare"] if not arg else arg == "on"
                elif cmd != "help":
                    print(f"unknown command /{cmd} -- type /help")
                    continue
            except ValueError:
                print(f"could not read '{arg}' -- type /help")
                continue
            print(HELP.format(**{**st, "compare": "on" if st["compare"] else "off"}) if cmd == "help" else
                  f"{DIM}gamma {st['gamma']}, temp {st['temp']}, tokens {st['tokens']}, until {st['until']}, "
                  f"compare {'on' if st['compare'] else 'off'}{RESET}")
            continue

        out = c.complete(text, st["gamma"], st["temp"], st["tokens"], st["until"], st["compare"], seed)
        seed += 1
        if out is None:
            continue
        spec, base = out
        print(f"{BOLD}model>{RESET} {text}{c.coloured(spec.tokens, spec.sources)}")
        c.report(spec, base, st["temp"], short=True)
        print()

    t = c.totals
    if t["lines"]:
        msg = f"\n{t['lines']} completions, {t['spec_tokens']} tokens: speculative {t['spec_tokens'] / t['spec_s']:.0f} tok/s"
        if t["base_s"]:
            sp = (t["spec_tokens"] / t["spec_s"]) / (t["base_tokens"] / t["base_s"])
            msg += f", normal {t['base_tokens'] / t['base_s']:.0f} tok/s, overall speedup {sp:.2f}x"
        print(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=list(config.PAIRS), default="scratch")
    ap.add_argument("--draft", default=None, help="plain / distilled (default: distilled if it exists)")
    ap.add_argument("--prompt", default="Once upon a time, there was a little girl named Lily.")
    ap.add_argument("--interactive", "-i", action="store_true", help="type prompts one after another")
    ap.add_argument("--until", choices=["sentence", "line", "none"], default=None,
                    help="where a completion stops (default: sentence in interactive mode, none otherwise)")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="default: 60 interactive, 150 otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    a.until = a.until or ("sentence" if a.interactive else "none")
    a.max_new_tokens = a.max_new_tokens or (60 if a.interactive else 150)
    if os.name == "nt":
        os.system("")                      # turns on colours in the Windows terminal

    c = Completer(a)
    if a.interactive:
        interactive(c, a)
        return
    out = c.complete(a.prompt, a.gamma, a.temperature, a.max_new_tokens, a.until, True, a.seed)
    if out is None:
        raise SystemExit("the prompt is empty")
    spec, base = out
    print(f"\npair {a.pair}, draft '{c.dname}', gamma {a.gamma}, temperature {a.temperature}\n")
    print(a.prompt + c.coloured(spec.tokens, spec.sources))
    c.report(spec, base, a.temperature)


if __name__ == "__main__":
    main()
