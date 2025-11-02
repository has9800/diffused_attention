from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from AdaptivePerHeadAttention import AdaptivePerHeadAttention


def _make_adaptive_attn(
    attn_module,
    adaptive_module: AdaptivePerHeadAttention,
    sink_indices: Optional[Iterable[int]],
):
    """
    Wrap the module-level `_attn` function so the softmax step is replaced with
    AdaptivePerHeadAttention while keeping the rest of the logic untouched.
    """

    scaling = getattr(attn_module, "scaling", 1.0 / max(adaptive_module.head_dim, 1))
    dropout_p = getattr(attn_module, "attention_dropout", 0.0)

    def patched_attn(query_states, key_states, value_states, attention_mask=None, **kwargs):
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        probs = adaptive_module(attn_scores, attention_mask=None, sink_indices=sink_indices)
        probs = F.dropout(probs, p=dropout_p, training=attn_module.training)

        attn_output = torch.matmul(probs, value_states)
        return attn_output, probs

    return patched_attn


def _get_projection_device(attn_module) -> torch.device:
    q_proj = getattr(attn_module, "q_proj", None)
    if q_proj is not None:
        weight = getattr(q_proj, "weight", None)
        if weight is not None and hasattr(weight, "device"):
            return weight.device
    for param in attn_module.parameters():
        return param.device
    return torch.device("cpu")


def patch_attention(
    model,
    mode: str = "adaptive_per_head",
    sink_indices: Optional[Iterable[int]] = (0,),
):
    """
    Patch the decoder self-attention of a LLaMA/Mistral-style model to use
    AdaptivePerHeadAttention. Only `_attn` is overridden so rotary embeddings,
    KV caching, and sliding windows continue to function as upstream.
    """
    if mode not in {"fixed_two_pool", "adaptive_per_head"}:
        raise ValueError(f"Unsupported mode {mode}")

    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise AttributeError("Expected model.model.layers to exist for patching.") from exc

    num_layers = len(layers)

    for layer_idx, decoder_layer in enumerate(layers):
        attn = decoder_layer.self_attn

        if not hasattr(attn, "_original_attn"):
            attn._original_attn = attn._attn

        if hasattr(attn, "_adaptive_module"):
            adaptive_module = attn._adaptive_module
        else:
            adaptive_module = AdaptivePerHeadAttention(
                dim=attn.hidden_size,
                num_heads=attn.num_heads,
                layer_idx=layer_idx,
                num_layers=num_layers,
                use_adaptive=(mode == "adaptive_per_head"),
                enable_two_pool=True,
            )
            device = _get_projection_device(attn)
            adaptive_module = adaptive_module.to(device)
            attn._adaptive_module = adaptive_module

        adaptive_module.use_adaptive = mode == "adaptive_per_head"
        adaptive_module.enable_two_pool = True

        attn._attn = _make_adaptive_attn(attn, adaptive_module, sink_indices)


def unpatch_attention(model):
    """
    Restore the original `_attn` functions if they were previously patched.
    """
    try:
        layers = model.model.layers
    except AttributeError:
        return

    for decoder_layer in layers:
        attn = decoder_layer.self_attn
        if hasattr(attn, "_original_attn"):
            attn._attn = attn._original_attn
            delattr(attn, "_original_attn")
        if hasattr(attn, "_adaptive_module"):
            delattr(attn, "_adaptive_module")
