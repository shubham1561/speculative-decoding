"""One small interface for both kinds of model.

The decoding loops only ever call these four things:

    cache  = lm.new_cache()
    logits = lm.forward(token_ids, cache)   # (len(token_ids), vocab), float32; appends to cache
    lm.crop(cache, n)                       # keep the first n cached tokens
    lm.cache_len(cache)

`OwnGPT` wraps the from-scratch model in gpt.py. `HFModel` wraps any HuggingFace
causal LM (GPT-2, Qwen, Pythia, ...). Nothing from HuggingFace's own speculative
decoding (`assistant_model`) is used anywhere.
"""
import torch


class OwnGPT:
    kind = "own"

    def __init__(self, model, name="gpt"):
        self.model = model.eval()
        self.name = name
        self.vocab_size = model.cfg.vocab_size
        self.max_len = model.cfg.block_size
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype

    def new_cache(self):
        return self.model.new_cache()

    @torch.no_grad()
    def forward(self, ids, cache):
        x = torch.as_tensor(ids, dtype=torch.long, device=self.device).view(1, -1)
        logits, _ = self.model(x, cache=cache)
        return logits[0].float()

    def crop(self, cache, n):
        cache.crop(n)

    def cache_len(self, cache):
        return cache.length

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())


class _HFCache:
    def __init__(self):
        self.c = None


class HFModel:
    kind = "hf"

    def __init__(self, model, name="hf"):
        self.model = model.eval()
        self.name = name
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self.max_len = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype

    def new_cache(self):
        from transformers import DynamicCache
        h = _HFCache()
        h.c = DynamicCache()
        return h

    @torch.no_grad()
    def forward(self, ids, cache):
        x = torch.as_tensor(ids, dtype=torch.long, device=self.device).view(1, -1)
        out = self.model(input_ids=x, past_key_values=cache.c, use_cache=True)
        cache.c = out.past_key_values
        return out.logits[0].float()

    def crop(self, cache, n):
        cur = self.cache_len(cache)
        if n < cur:
            cache.c.crop(n - cur)          # negative = remove that many (works in transformers 4.x and 5.x)

    def cache_len(self, cache):
        return 0 if cache.c is None else int(cache.c.get_seq_length())

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())
