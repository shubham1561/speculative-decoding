"""Step 1: download TinyStories and turn it into token files.

    python prepare_data.py                       # all of TinyStories (~470M tokens, ~10-20 min)
    python prepare_data.py --max_stories 300000  # a smaller slice, much faster
    python prepare_data.py --source my_text.txt  # your own text instead

Writes data/tinystories/train.bin, val.bin and meta.json (see config.DATA_DIR).
"""
import argparse
import os
import re
import time

import config
from specdec.data import get_tokenizer, write_tokens, write_meta


def encode_stories(tok, stories):
    if hasattr(tok, "encode_batch"):
        ids = tok.encode_batch(stories)
    else:
        ids = [tok.encode(s) for s in stories]
    for t in ids:
        t.append(tok.eot)                       # end-of-text between stories
    return ids


def write_split(tok, stories, path, chunk=20000, label=""):
    total, first = 0, True
    buf = []
    t0 = time.time()
    for i, s in enumerate(stories, 1):
        s = s.strip()
        if s:
            buf.append(s)
        if len(buf) >= chunk:
            total += write_tokens(path, encode_stories(tok, buf), append=not first)
            first, buf = False, []
            print(f"  {label}: {i:,} stories, {total:,} tokens  ({time.time() - t0:.0f} s)", flush=True)
    if buf or first:
        total += write_tokens(path, encode_stories(tok, buf), append=not first)
    print(f"  {label}: done, {total:,} tokens")
    return total


def prepare_from_lists(train_stories, val_stories, tokenizer_name, out_dir, source):
    """Used by this script and by smoke_test.py."""
    os.makedirs(out_dir, exist_ok=True)
    tok = get_tokenizer(tokenizer_name)
    counts = {
        "train": write_split(tok, train_stories, os.path.join(out_dir, "train.bin"), label="train"),
        "val": write_split(tok, val_stories, os.path.join(out_dir, "val.bin"), label="val"),
    }
    meta = write_meta(out_dir, tok, counts, source)
    print(f"saved to {out_dir}: {meta}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="tinystories", help="'tinystories' or a path to a .txt file")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out", default=config.DATA_DIR)
    ap.add_argument("--max_stories", type=int, default=0, help="0 = all")
    ap.add_argument("--val_fraction", type=float, default=0.01, help="only for a .txt source")
    a = ap.parse_args()

    if a.source == "tinystories":
        from datasets import load_dataset
        print("downloading TinyStories from the HuggingFace Hub (about 1 GB, only the first time)...")
        ds = load_dataset("roneneldan/TinyStories")
        train = ds["train"]
        if a.max_stories:
            train = train.select(range(min(a.max_stories, len(train))))
        val = ds["validation"]
        prepare_from_lists((r["text"] for r in train), (r["text"] for r in val), a.tokenizer, a.out,
                           f"roneneldan/TinyStories ({len(train):,} train stories)")
    else:
        with open(a.source, encoding="utf-8") as f:
            text = f.read()
        parts = [p for p in re.split(r"<\|endoftext\|>|\n\s*\n", text) if p.strip()]
        if a.max_stories:
            parts = parts[: a.max_stories]
        n_val = max(1, int(len(parts) * a.val_fraction))
        prepare_from_lists(parts[n_val:], parts[:n_val], a.tokenizer, a.out, os.path.basename(a.source))


if __name__ == "__main__":
    main()
