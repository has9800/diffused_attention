import os
os.environ['HF_HOME'] = '/data/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/data/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/data/datasets_cache'
os.makedirs('/data/huggingface_cache', exist_ok=True)
os.makedirs('/data/datasets_cache', exist_ok=True)
os.makedirs('/data/cache', exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from datasets import load_dataset
import math
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import nltk

nltk.download('punkt', quiet=True)

# ============================================================================
# ADAPTIVE ATTENTION CLASS
# ============================================================================

class AdaptivePerHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_idx: int = 0,
        num_layers: int = 24,
        use_adaptive: bool = True,
        enable_two_pool: bool = True,
    ) -> None:
        super().__init__()
        if dim % max(num_heads, 1) != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.num_layers = max(num_layers, 1)
        self.use_adaptive = use_adaptive
        self.enable_two_pool = enable_two_pool
        self.head_dim = dim // max(num_heads, 1)
        depth = layer_idx / max(self.num_layers - 1, 1)
        self.register_buffer("layer_depth", torch.tensor(depth, dtype=torch.float32))

    def get_adaptive_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_adaptive:
            epsilon = torch.full((self.num_heads,), 0.02)
            temperature = torch.ones(self.num_heads)
            return epsilon, temperature
        layer_depth = float(self.layer_depth.item())
        epsilon = torch.full((self.num_heads,), 0.01 + 0.04 * layer_depth)
        temperature = torch.full((self.num_heads,), 1.0 + 0.8 * layer_depth)
        return epsilon, temperature

    @staticmethod
    def _prepare_epsilon(epsilon: torch.Tensor, num_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        epsilon = epsilon.to(device=device, dtype=dtype)
        epsilon = epsilon.view(1, num_heads, 1, 1)
        epsilon = epsilon.clamp_min(1e-6).clamp_max(1.0 - 1e-6)
        return epsilon

    def two_pool_renormalization(
        self,
        weights: torch.Tensor,
        epsilon: torch.Tensor,
        sink_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if not sink_indices:
            return weights
        sink_indices = list(sink_indices)
        src_len = weights.size(-1)
        sink_mask = torch.zeros(src_len, device=weights.device, dtype=torch.bool)
        sink_mask[sink_indices] = True
        content_mask = ~sink_mask
        sink_mass = weights[..., sink_mask].sum(dim=-1, keepdim=True, dtype=torch.float32)
        content_mass = weights[..., content_mask].sum(dim=-1, keepdim=True, dtype=torch.float32)
        epsilon_expanded = self._prepare_epsilon(epsilon, self.num_heads, weights.device, weights.dtype)
        output = torch.zeros_like(weights)
        output[..., content_mask] = (
            weights[..., content_mask]
            * (1.0 - epsilon_expanded)
            / (content_mass + 1e-9)
        )
        output[..., sink_mask] = (
            weights[..., sink_mask]
            * epsilon_expanded
            / (sink_mass + 1e-9)
        )
        return output

    def forward(
        self,
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sink_indices: Optional[Iterable[int]] = (0,),
    ) -> torch.Tensor:
        if attention_mask is not None:
            scores = scores + attention_mask
        epsilon, temperature = self.get_adaptive_params()
        temperature = temperature.to(device=scores.device, dtype=scores.dtype)
        temperature = temperature.view(1, self.num_heads, 1, 1).clamp_min(1e-4)
        scores = scores / temperature
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=scores.dtype)
        if self.enable_two_pool:
            probs = self.two_pool_renormalization(probs, epsilon, sink_indices=sink_indices)
        return probs


# ============================================================================
# CUSTOM QWEN2 ATTENTION (FIXED)
# ============================================================================

class CustomQwen2Attention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # EXPLICITLY SET ATTRIBUTES
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size
        
        self.custom_softmax = AdaptivePerHeadAttention(
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            layer_idx=layer_idx if layer_idx is not None else 0,
            num_layers=config.num_hidden_layers,
            use_adaptive=True,
            enable_two_pool=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value, key_states], dim=2)
            value_states = torch.cat([past_key_value, value_states], dim=2)

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = self.custom_softmax(attn_weights, attention_mask=None)

        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ============================================================================
# PATCH FUNCTION
# ============================================================================

def patch_model_with_custom_attention(model, device):
    """Replace all attention layers with custom adaptive attention."""
    dtype = model.dtype
    for i, layer in enumerate(model.model.layers):
        custom_attn = CustomQwen2Attention(model.config, layer_idx=i)
        custom_attn = custom_attn.to(dtype=dtype, device=device)
        custom_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
        layer.self_attn = custom_attn
    return model


# ============================================================================
# BENCHMARK: CODE-IN-CONTEXT
# ============================================================================

def eval_code_in_context(model, tokenizer, context_length=8192, num_examples=20, device='cuda'):
    """
    Task: Given first half of long Python file, predict second half.
    Metric: BLEU score (higher is better).
    
    Why: Tests if adaptive attention helps with long code understanding.
    """
    
    try:
        dataset = load_dataset("code_search_net", "python", split="test", cache_dir="/tmp/cache")
    except:
        print("  [Warning] CodeSearchNet unavailable, using alternative dataset")
        return []
    
    results = []
    bleu_scores = []
    smoothing_fn = SmoothingFunction().method1
    
    model.eval()
    
    for idx, example in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
        code = example['content']
        
        # Skip if code too short
        code_tokens = tokenizer(code, return_tensors="pt").input_ids.shape
        if code_tokens < context_length + 100:
            continue
        
        # Split: first half = context, second half = target
        split_point = len(code) // 2
        context = code[:split_point]
        target = code[split_point:]
        
        try:
            # Tokenize context
            inputs = tokenizer(context[:context_length * 3], return_tensors="pt", truncation=True, max_length=context_length).to(device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0
                )
            
            # Decode prediction
            prediction = tokenizer.decode(outputs[inputs.input_ids.size(1):], skip_special_tokens=True)
            
            # Compute BLEU
            reference = [target.split()]
            candidate = prediction.split()
            bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing_fn)
            bleu_scores.append(bleu)
            
            if idx % 5 == 0:
                print(f"    Sample {idx}: BLEU = {bleu:.4f}")
        
        except RuntimeError as e:
            print(f"    [Warning] OOM or error on sample {idx}: {str(e)[:50]}")
            continue
    
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        results.append({
            'context_length': context_length,
            'avg_bleu': avg_bleu,
            'num_samples': len(bleu_scores)
        })
        print(f"  Average BLEU: {avg_bleu:.4f}")
    
    return results


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmarks(
    model_name="Qwen/Qwen2.5-7B",
    device='cuda',
    context_length=8192,
    num_examples=20,
    output_dir="/tmp/code_benchmark_results"
):
    """Run code-in-context benchmark on baseline and adaptive attention."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CODE-IN-CONTEXT BENCHMARK")
    print(f"Model: {model_name}")
    print(f"Context Length: {context_length}")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).to(device)
    
    # ---- BASELINE EVALUATION ----
    print("\n=== BASELINE EVALUATION ===")
    print("Code-in-context (standard attention)...")
    baseline_results = eval_code_in_context(
        model,
        tokenizer,
        context_length=context_length,
        num_examples=num_examples,
        device=device
    )
    
    # ---- PATCH & ADAPTIVE EVALUATION ----
    print("\n=== PATCHING WITH CUSTOM ATTENTION ===")
    model = patch_model_with_custom_attention(model, device)
    
    print("\n=== ADAPTIVE ATTENTION EVALUATION ===")
    print("Code-in-context (adaptive attention)...")
    adaptive_results = eval_code_in_context(
        model,
        tokenizer,
        context_length=context_length,
        num_examples=num_examples,
        device=device
    )
    
    # ---- SAVE RESULTS ----
    print("\n=== SAVING RESULTS ===")
    
    if baseline_results:
        pd.DataFrame(baseline_results).to_csv(f"{output_dir}/code_baseline.csv", index=False)
        print(f"✓ Baseline results: {baseline_results['avg_bleu']:.4f}")
    
    if adaptive_results:
        pd.DataFrame(adaptive_results).to_csv(f"{output_dir}/code_adaptive.csv", index=False)
        print(f"✓ Adaptive results: {adaptive_results['avg_bleu']:.4f}")
    
    # Compute improvement
    if baseline_results and adaptive_results:
        baseline_bleu = baseline_results['avg_bleu']
        adaptive_bleu = adaptive_results['avg_bleu']
        improvement = ((adaptive_bleu - baseline_bleu) / baseline_bleu * 100) if baseline_bleu > 0 else 0
        print(f"\n🎯 IMPROVEMENT: {improvement:+.2f}%")
    
    print(f"\nResults saved to {output_dir}")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    run_benchmarks(
        model_name="Qwen/Qwen2.5-7B",
        device='cuda',
        context_length=8192,
        num_examples=20,
        output_dir="/tmp/code_benchmark_results"
    )
