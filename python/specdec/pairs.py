"""Load target/draft models (own checkpoints or HuggingFace) behind the same adapter."""
import os

import torch

from .adapters import OwnGPT, HFModel
from .gpt import load_gpt


def load_hf_model(name_or_path, device, dtype):
    import transformers
    from transformers import AutoModelForCausalLM
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"       # the argument was renamed
    m = AutoModelForCausalLM.from_pretrained(name_or_path, **{key: dtype})
    return m.to(device=device, dtype=dtype).eval()


def inference_dtype(device, preferred=None):
    """fp16 on the GPU by default: more precise than bf16, so greedy speculative output matches
    normal decoding more often. A pair can ask for "float32" (small models: exact greedy match, and
    they are limited by overhead, not maths) or "bfloat16" (Qwen is trained in bf16) in config.PAIRS."""
    if device.type != "cuda" or preferred == "float32":
        return torch.float32
    if preferred == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def exists(spec):
    kind, where = spec
    if kind == "own":
        return os.path.exists(where)
    if os.path.isabs(where):
        return os.path.isdir(where)          # a local folder (e.g. a distilled draft)
    return True                              # a Hub id like "gpt2-large": downloaded on first use


def load_lm(spec, device, dtype=None, name=None):
    kind, where = spec
    dtype = dtype or inference_dtype(device)
    if kind == "own":
        return OwnGPT(load_gpt(where, device, dtype), name or os.path.basename(where))
    return HFModel(load_hf_model(where, device, dtype), name or where)
