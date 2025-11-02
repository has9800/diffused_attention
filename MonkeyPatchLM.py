import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

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


def _make_adaptive_forward(
    attn_module,
    adaptive_module: AdaptivePerHeadAttention,
    sink_indices: Optional[Iterable[int]],
):
    """
    Fallback wrapper for attention modules that do not expose an `_attn` helper
    (e.g., MistralAttention in newer Transformers releases).
    """

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

        kv_seq_len = key_states.size(-2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            kv_seq_len += past_k.size(-2)
            key_states = torch.cat([past_k, key_states], dim=-2)
            value_states = torch.cat([past_v, value_states], dim=-2)

        present_key_value = (key_states, value_states) if use_cache else None

        rotary_kwargs = {}
        if position_ids is not None:
            rotary_kwargs["position_ids"] = position_ids
        if cache_position is not None:
            rotary_kwargs["position_ids"] = cache_position
        elif "cache_position" in kwargs and kwargs["cache_position"] is not None:
            rotary_kwargs["position_ids"] = kwargs["cache_position"]

        cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, **rotary_kwargs)

        key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
        value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(attn_module.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        probs = adaptive_module(attn_scores, attention_mask=None, sink_indices=sink_indices)
        dropout_p = getattr(attn_module, "attention_dropout", 0.0)
        probs = F.dropout(probs, p=dropout_p, training=attn_module.training)

        attn_output = torch.matmul(probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, attn_module.hidden_size)
        attn_output = attn_module.o_proj(attn_output)

        if not output_attentions:
            probs = None

        return attn_output, probs, present_key_value

    return forward


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

        if hasattr(attn, "_attn"):
            if not hasattr(attn, "_original_attn"):
                attn._original_attn = attn._attn
            attn._attn = _make_adaptive_attn(attn, adaptive_module, sink_indices)
        else:
            if not hasattr(attn, "_original_forward"):
                attn._original_forward = attn.forward
            attn.forward = _make_adaptive_forward(attn, adaptive_module, sink_indices)


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
        if hasattr(attn, "_original_forward"):
            attn.forward = attn._original_forward
            delattr(attn, "_original_forward")
