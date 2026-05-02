"""Model creation factory for supported architectures."""

import torch
import torch.nn as nn


_MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model architecture."""
    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def create_model(name, dtype=torch.bfloat16, rope_scaling=None):
    """Create a model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[name]
    return cls(dtype=dtype, rope_scaling=rope_scaling)


@register_model("llama-7b")
@register_model("llama-13b")
@register_model("llama-70b")
class LlamaWrapper(nn.Module):
    """Wrapper around HuggingFace Llama models with context extension."""

    def __init__(self, dtype=torch.bfloat16, rope_scaling=None):
        super().__init__()
        self.dtype = dtype
        self.rope_scaling = rope_scaling
        self.model = None

    def forward(self, input_ids, attention_mask=None):
        if self.model is None:
            self._lazy_load()
        return self.model(input_ids, attention_mask=attention_mask)

    def _lazy_load(self):
        try:
            from transformers import LlamaForCausalLM
            name = getattr(self, "_registry_name", "llama-70b")
            hf_map = {
                "llama-7b": "meta-llama/Llama-2-7b-hf",
                "llama-13b": "meta-llama/Llama-2-13b-hf",
                "llama-70b": "meta-llama/Llama-3-70b",
            }
            self.model = LlamaForCausalLM.from_pretrained(
                hf_map[name],
                torch_dtype=self.dtype,
                rope_scaling=self.rope_scaling,
                device_map="auto",
            )
        except ImportError:
            raise RuntimeError("transformers not installed")


@register_model("deepseek-v3")
class DeepSeekWrapper(nn.Module):
    """Wrapper around DeepSeek V3 model."""

    def __init__(self, dtype=torch.bfloat16, rope_scaling=None):
        super().__init__()
        self.dtype = dtype
        self.rope_scaling = rope_scaling
        self.model = None

    def forward(self, input_ids, attention_mask=None):
        if self.model is None:
            self._lazy_load()
        return self.model(input_ids, attention_mask=attention_mask)

    def _lazy_load(self):
        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-v3",
                torch_dtype=self.dtype,
                rope_scaling=self.rope_scaling,
                device_map="auto",
            )
        except ImportError:
            raise RuntimeError("transformers not installed")