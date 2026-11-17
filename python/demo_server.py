"""Browser demo: type a prompt, watch normal and speculative decoding race.

    python demo_server.py              # opens http://127.0.0.1:8000 in your browser
    python demo_server.py --port 8080 --no-browser

Or double-click START_DEMO.bat in the repo folder.

The server only listens on your own computer (127.0.0.1). It loads a model pair
once (GPT-2 large + DistilGPT-2 by default, or your own models), then for every
prompt runs speculative decoding and normal decoding on the same input, with the
GPU warmed up before each timed run, and sends back the text, which tokens came
from the draft, and when each one was produced. The page replays both runs at
the speed they were measured.

Stop it with Ctrl+C, or close its window.
"""
import argparse
import json
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import config
from generate import stop_token_ids
from specdec.data import get_tokenizer
from specdec.decoding import autoregressive_generate, speculative_generate, warm_up
from specdec.pairs import exists, inference_dtype, load_lm
from specdec.training import pick_device

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "demo", "index.html")

LABELS = {
    "gpt2": "GPT-2 large (774M) + DistilGPT-2 (82M)",
    "scratch": "My own models: 51M target + 15M draft (stories only)",
    "qwen": "Qwen2.5 1.5B + 0.5B (downloads ~4 GB the first time)",
}
SHOWN = ["gpt2", "scratch"]          # pairs offered on the page (add "qwen" to try it)


class Engine:
    """Loads pairs on demand and runs one comparison at a time."""

    def __init__(self, device):
        self.device = device
        self.loaded = {}             # pair -> {"tok", "target", "drafts": {name: lm}, "stops": {}}
        self.loading = None          # pair being loaded right now
        self.error = None
        self.lock = threading.Lock()     # one GPU job at a time
        self.state_lock = threading.Lock()

    # ---- loading ----------------------------------------------------------
    def available_pairs(self):
        out = []
        for name in SHOWN:
            pair = config.PAIRS.get(name)
            if not pair or not exists(pair["target"]):
                continue
            drafts = [d for d, spec in pair["drafts"].items() if exists(spec)]
            if drafts:
                out.append({"name": name, "label": LABELS.get(name, name), "drafts": drafts,
                            "loaded": name in self.loaded})
        return out

    def start_loading(self, name):
        with self.state_lock:
            if name in self.loaded or self.loading == name:
                return
            if self.loading:
                raise RuntimeError(f"still loading {self.loading}, try again in a moment")
            self.loading, self.error = name, None
        threading.Thread(target=self._load, args=(name,), daemon=True).start()

    def _load(self, name):
        try:
            pair = config.PAIRS[name]
            dtype = inference_dtype(self.device, pair.get("dtype"))
            print(f"loading pair '{name}' on {self.device} ...", flush=True)
            tok = get_tokenizer(pair["tokenizer"])
            target = load_lm(pair["target"], self.device, dtype, "target")
            drafts = {d: load_lm(spec, self.device, dtype, d) for d, spec in pair["drafts"].items() if exists(spec)}
            entry = {"tok": tok, "target": target, "drafts": drafts, "stops": {}}
            for until in ("sentence", "line"):       # look these up now so the first prompt is quick
                entry["stops"][until] = stop_token_ids(tok, until)
            entry["stops"]["none"] = set()
            with self.lock:                          # one-time GPU warm-up (kernels, memory)
                warm = tok.encode("Once upon a time") or [0]
                for d in drafts.values():
                    speculative_generate(target, d, warm, 8, 2)
                autoregressive_generate(target, warm, 8)
            self.loaded[name] = entry
            print(f"pair '{name}' ready", flush=True)
        except Exception as e:                       # shown on the page
            traceback.print_exc()
            self.error = f"could not load '{name}': {e}"
        finally:
            self.loading = None

    def status(self):
        gpu = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU (no GPU found)"
        return {"device": gpu, "pairs": self.available_pairs(), "loading": self.loading, "error": self.error}

    # ---- one comparison --------------------------------------------------
    def complete(self, req):
        name = req.get("pair", "gpt2")
        if name not in self.loaded:
            raise RuntimeError(f"pair '{name}' is not loaded yet")
        e = self.loaded[name]
        tok, target = e["tok"], e["target"]
        draft_name = req.get("draft") or ("distilled" if "distilled" in e["drafts"] else next(iter(e["drafts"])))
        draft = e["drafts"][draft_name]
        gamma = max(0, min(12, int(req.get("gamma", 4))))
        temperature = max(0.0, min(2.0, float(req.get("temperature", 0.0))))
        top_p = max(0.05, min(1.0, float(req.get("top_p", 1.0))))
        max_new = max(1, min(256, int(req.get("max_new_tokens", 64))))
        until = req.get("until", "sentence")
        seed = int(req.get("seed", 0))
        text = str(req.get("prompt", ""))[:2000]
        prompt = tok.encode(text)
        if not prompt:
            raise RuntimeError("type a few words first")
        room = min(m for m in (target.max_len, draft.max_len) if m) - max_new
        if room < 1:
            max_new = min(target.max_len, draft.max_len) - 1
            room = 1
        prompt = prompt[-room:]
        kw = dict(temperature=temperature, top_p=top_p, eos_id=getattr(tok, "eot", None),
                  stop_ids=e["stops"].get(until, set()))

        with self.lock:
            warm_up([target, draft])
            g = torch.Generator(device=self.device).manual_seed(seed)
            spec = speculative_generate(target, draft, prompt, max_new, gamma, generator=g, **kw)
            warm_up([target])
            g = torch.Generator(device=self.device).manual_seed(seed)
            base = autoregressive_generate(target, prompt, max_new, generator=g, **kw)

        s, b = spec.stats, base.stats
        return {
            "pair": name, "draft": draft_name, "gamma": gamma, "temperature": temperature,
            "prompt_text": tok.decode(prompt),
            "spec": {
                "pieces": pieces(tok, prompt, spec.tokens),
                "sources": spec.sources,
                "rounds": round_ids(s.accepted_per_round, len(spec.tokens)),
                "timeline": s.timeline, "seconds": s.seconds, "tokens": s.new_tokens,
                "tok_per_s": s.tokens_per_second, "target_calls": s.target_calls,
                "draft_calls": s.draft_calls, "proposed": s.proposed, "accepted": s.accepted,
                "accept_rate": s.acceptance_rate if s.proposed else None,
                "free_tokens": spec.sources.count("draft"),
            },
            "normal": {
                "pieces": pieces(tok, prompt, base.tokens),
                "timeline": b.timeline, "seconds": b.seconds, "tokens": b.new_tokens,
                "tok_per_s": b.tokens_per_second, "target_calls": b.target_calls,
            },
            "speedup": (s.tokens_per_second / b.tokens_per_second) if b.tokens_per_second else None,
            "same_text": spec.tokens == base.tokens,
        }


def pieces(tok, prompt, tokens):
    """The text each new token adds. Decoding prefixes keeps characters split over tokens intact."""
    out, prev = [], tok.decode(prompt)
    seq = list(prompt)
    for t in tokens:
        seq.append(t)
        cur = tok.decode(seq)
        out.append(cur[len(prev):] if cur.startswith(prev) else tok.decode([t]))
        prev = cur
    return out


def round_ids(accepted_per_round, n_tokens):
    """Which round each token belongs to (a round = n accepted + 1 from the target)."""
    ids = []
    for r, n in enumerate(accepted_per_round):
        ids += [r] * (n + 1)
    return ids[:n_tokens]


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):          # keep the window quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._send(200, engine.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/load":
                    engine.start_loading(req.get("pair", "gpt2"))
                    self._send(200, engine.status())
                elif self.path == "/api/complete":
                    self._send(200, engine.complete(req))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                if not isinstance(e, RuntimeError):
                    traceback.print_exc()
                self._send(400, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pair", default=None, help="pair to load first (default: gpt2 if available)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    engine = Engine(pick_device(a.device))
    pairs = [p["name"] for p in engine.available_pairs()]
    if not pairs:
        raise SystemExit("no trained model pair found -- train one first (GUIDE_2 / GUIDE_3)")
    first = a.pair if a.pair in pairs else pairs[0]

    server = None
    for port in range(a.port, a.port + 10):          # use the next free port if 8000 is taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"ports {a.port}-{a.port + 9} are all in use")
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    engine.start_loading(first)
    print("=" * 64)
    print(f" Speculative decoding demo:  {url}")
    print(f" device: {engine.status()['device']}")
    print(f" loading '{first}' -- the page shows when it is ready (~10-30 s)")
    print(" Stop with Ctrl+C or close this window.")
    print("=" * 64, flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
