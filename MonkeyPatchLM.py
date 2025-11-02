import math
import inspect
from typing import Iterable, Optional, NamedTuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # type: ignore
except Exception:  # pragma: no cover
    LlamaRotaryEmbedding = None  # type: ignore

from AdaptivePerHeadAttention import AdaptivePerHeadAttention


class AttentionMeta(NamedTuple):
    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int


def _make_adaptive_attn(
    attn_module,
    adaptive_module: AdaptivePerHeadAttention,
    sink_indices: Optional[Iterable[int]],
    meta: AttentionMeta,
):
    """
    Wrap the module-level `_attn` function so the softmax step is replaced with
    AdaptivePerHeadAttention while keeping the rest of the logic untouched.
    """

    scaling = getattr(attn_module, "scaling", 1.0 / math.sqrt(max(meta.head_dim, 1)))
    dropout_p = getattr(attn_module, "attention_dropout", getattr(attn_module, "dropout", 0.0))

    def patched_attn(query_states, key_states, value_states, attention_mask=None, **kwargs):
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                neg = -1e4 if attn_scores.dtype == torch.float16 else -1e9
                attn_scores = attn_scores.masked_fill(~attention_mask, neg)
            else:
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
    meta: AttentionMeta,
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

        query_states = query_states.view(bsz, q_len, meta.num_heads, meta.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, meta.num_key_value_heads, meta.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, meta.num_key_value_heads, meta.head_dim).transpose(1, 2)

        kv_seq_len = key_states.size(-2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            kv_seq_len += past_k.size(-2)
            key_states = torch.cat([past_k, key_states], dim=-2)
            value_states = torch.cat([past_v, value_states], dim=-2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Apply rotary embeddings using the module's provider if available.
        # Otherwise, lazily construct a LlamaRotaryEmbedding as a fallback.
        rotary_kwargs = {}
        if position_ids is not None:
            rotary_kwargs["position_ids"] = position_ids
        if cache_position is not None:
            rotary_kwargs["position_ids"] = cache_position
        elif "cache_position" in kwargs and kwargs["cache_position"] is not None:
            rotary_kwargs["position_ids"] = kwargs["cache_position"]

        cos = sin = None
        if hasattr(attn_module, "rotary_emb") and callable(getattr(attn_module, "rotary_emb")):
            try:
                cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)
            except TypeError:
                # Some variants expect only seq_len.
                cos, sin = attn_module.rotary_emb(value_states, kv_seq_len)
        if cos is None or sin is None:
            rope = getattr(attn_module, "_adaptive_rotary", None)
            if rope is None and LlamaRotaryEmbedding is not None:
                try:
                    sig = inspect.signature(LlamaRotaryEmbedding.__init__)
                    if "config" in sig.parameters:
                        cfg = getattr(attn_module, "config", getattr(getattr(attn_module, "_config", None), "config", None))
                        if cfg is not None:
                            rope = LlamaRotaryEmbedding(cfg)
                    elif "dim" in sig.parameters:
                        base = getattr(attn_module, "rotary_emb_base", getattr(attn_module, "rope_theta", 10000))
                        rope = LlamaRotaryEmbedding(dim=meta.head_dim, base=base)
                    else:
                        rope = LlamaRotaryEmbedding(meta.head_dim)
                except Exception:
                    rope = None
                if rope is not None:
                    attn_module._adaptive_rotary = rope
            if rope is not None:
                cos, sin = rope(value_states, seq_len=kv_seq_len)

        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, **rotary_kwargs)

        key_states = repeat_kv(key_states, meta.num_key_value_groups)
        value_states = repeat_kv(value_states, meta.num_key_value_groups)

        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(meta.head_dim)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                neg = -1e4 if attn_scores.dtype == torch.float16 else -1e9
                attn_scores = attn_scores.masked_fill(~attention_mask, neg)
            else:
                attn_scores = attn_scores + attention_mask

        probs = adaptive_module(attn_scores, attention_mask=None, sink_indices=sink_indices)
        dropout_p = getattr(attn_module, "attention_dropout", getattr(attn_module, "dropout", 0.0))
        probs = F.dropout(probs, p=dropout_p, training=attn_module.training)

        attn_output = torch.matmul(probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, meta.hidden_size)
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


def _infer_attention_meta(attn_module, model_config) -> AttentionMeta:
    hidden_size = getattr(attn_module, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        q_proj = getattr(attn_module, "q_proj", None)
        if q_proj is not None and hasattr(q_proj, "out_features"):
            hidden_size = q_proj.out_features
        elif q_proj is not None and hasattr(q_proj, "weight"):
            hidden_size = q_proj.weight.shape[0]
    if hidden_size is None:
        raise AttributeError("Unable to infer attention hidden size for adaptive patching.")

    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn_module, "n_heads", None)
    if num_heads is None:
        num_heads = getattr(model_config, "num_attention_heads", None)
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None and num_heads:
        head_dim = hidden_size // num_heads
    if head_dim is None:
        if hasattr(model_config, "head_dim"):
            head_dim = model_config.head_dim
    if num_heads is None and head_dim:
        num_heads = hidden_size // head_dim
    if num_heads is None or head_dim is None:
        raise AttributeError("Unable to infer attention head configuration for adaptive patching.")

    num_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(attn_module, "num_kv_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(model_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_heads

    num_groups = getattr(attn_module, "num_key_value_groups", None)
    if num_groups is None:
        num_groups = getattr(model_config, "num_key_value_groups", None)
    if num_groups is None and num_kv_heads:
        num_groups = max(num_heads // max(num_kv_heads, 1), 1)
    if num_groups is None:
        num_groups = 1

    return AttentionMeta(
        hidden_size=int(hidden_size),
        num_heads=int(num_heads),
        num_key_value_heads=int(num_kv_heads),
        num_key_value_groups=int(num_groups),
        head_dim=int(head_dim),
    )


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
    model_config = getattr(model, "config", None)

    for layer_idx, decoder_layer in enumerate(layers):
        attn = decoder_layer.self_attn

        if hasattr(attn, "_adaptive_module"):
            adaptive_module = attn._adaptive_module
        else:
            meta = _infer_attention_meta(attn, model_config)
            adaptive_module = AdaptivePerHeadAttention(
                dim=meta.hidden_size,
                num_heads=meta.num_heads,
                layer_idx=layer_idx,
                num_layers=num_layers,
                use_adaptive=(mode == "adaptive_per_head"),
                enable_two_pool=True,
            )
            device = _get_projection_device(attn)
            adaptive_module = adaptive_module.to(device)
            attn._adaptive_module = adaptive_module
            attn._adaptive_meta = meta
        meta = getattr(attn, "_adaptive_meta", None)
        if meta is None:
            meta = _infer_attention_meta(attn, model_config)
            attn._adaptive_meta = meta

        adaptive_module.use_adaptive = mode == "adaptive_per_head"
        adaptive_module.enable_two_pool = True

        if hasattr(attn, "_attn"):
            if not hasattr(attn, "_original_attn"):
                attn._original_attn = attn._attn
            attn._attn = _make_adaptive_attn(attn, adaptive_module, sink_indices, meta)
        else:
            if not hasattr(attn, "_original_forward"):
                attn._original_forward = attn.forward
            attn.forward = _make_adaptive_forward(attn, adaptive_module, sink_indices, meta)


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
        if hasattr(attn, "_adaptive_meta"):
            delattr(attn, "_adaptive_meta")
