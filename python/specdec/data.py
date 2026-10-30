"""Tokenizers and the token files used for training.

Data lives as flat uint16 token files (train.bin / val.bin), with a meta.json
next to them. Stories are separated by the end-of-text token.
"""
import json
import os

import numpy as np
import torch


class ByteTokenizer:
    """Offline tokenizer (UTF-8 bytes + 1 end token). Used by the smoke test."""
    name = "bytes"
    vocab_size = 257
    eot = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    """GPT-2 BPE via tiktoken. Same token ids as HuggingFace's GPT-2 tokenizer."""
    name = "gpt2"
    vocab_size = 50257
    eot = 50256

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def encode_batch(self, texts):
        return self.enc.encode_ordinary_batch(texts)

    def decode(self, ids):
        return self.enc.decode(list(ids))


class HFTokenizer:
    """Any HuggingFace tokenizer (used for the Qwen pair)."""

    def __init__(self, repo):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.name = repo
        self.vocab_size = len(self.tok)
        self.eot = self.tok.eos_token_id

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True)


def get_tokenizer(name):
    if name == "bytes":
        return ByteTokenizer()
    if name == "gpt2":
        return GPT2Tokenizer()
    return HFTokenizer(name)


def write_tokens(path, token_lists, append=False):
    arr = np.concatenate([np.asarray(t, dtype=np.uint16) for t in token_lists]) if token_lists else np.zeros(0, np.uint16)
    with open(path, "ab" if append else "wb") as f:
        arr.tofile(f)
    return len(arr)


def write_meta(data_dir, tokenizer, counts, source):
    meta = {"tokenizer": tokenizer.name, "vocab_size": tokenizer.vocab_size, "eot": tokenizer.eot,
            "tokens": counts, "source": source}
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def read_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_split(data_dir, split):
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run prepare_data.py first (GUIDE_2, step 1)")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device, rng: np.random.Generator):
    """Random windows of block_size+1 tokens -> (inputs, targets)."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def story_prompts(data_dir, n, prompt_tokens, eot, seed=0):
    """Take the first `prompt_tokens` tokens of n validation stories."""
    data = load_split(data_dir, "val")
    starts = np.flatnonzero(np.asarray(data[: min(len(data), 5_000_000)]) == eot) + 1
    rng = np.random.default_rng(seed)
    rng.shuffle(starts)
    prompts = []
    for s in starts:
        p = data[s:s + prompt_tokens].astype(np.int64).tolist()
        if len(p) == prompt_tokens and eot not in p:
            prompts.append(p)
        if len(prompts) == n:
            break
    return prompts
