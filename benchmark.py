import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Iterable, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from datasets import load_dataset
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import nltk

nltk.download('punkt', quiet=True)

# ============================================================================
# ADAPTIVE ATTENTION CLASS (Keep as-is, it's correct)
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
# CUSTOM ATTENTION CLASS (FIXED)
# ============================================================================

class CustomQwen2Attention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
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

        # FIX: Use custom softmax with adaptive parameters
        attn_probs = self.custom_softmax(attn_weights, attention_mask=None)  # Already applied above

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
        # Copy weights from original
        custom_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
        layer.self_attn = custom_attn
    return model


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def eval_language_modeling(model, tokenizer, context_lengths, device, batch_size=1, max_model_len=32768, stride=512):
    """Evaluate perplexity across context lengths."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir="/tmp/cache")
    results = []

    for context_len in context_lengths:
        effective_context = min(context_len, max_model_len)
        total_nll = 0.0
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction='sum')

        for idx, text in enumerate(dataset["text"]):
            if not text.strip():
                continue
            
            try:
                encodings = tokenizer(text, return_tensors="pt").to(device)
                seq_len = encodings.input_ids.size(1)
                if seq_len < 2:
                    continue

                nlls = []
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + effective_context, seq_len)
                    if end_loc - begin_loc <= 1:
                        continue
                    input_ids = encodings.input_ids[:, begin_loc:end_loc]
                    if input_ids.size(1) > max_model_len:
                        input_ids = input_ids[:, -max_model_len:]

                    with torch.no_grad():
                        outputs = model(input_ids)
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        nlls.append(loss)
                        total_tokens += shift_labels.numel()

                if nlls:
                    doc_nll = torch.stack(nlls).sum().item()
                    total_nll += doc_nll
            except RuntimeError as e:
                print(f"  [Warning] OOM or error on sample {idx}: {str(e)[:50]}")
                continue

        if total_tokens > 0:
            avg_loss = total_nll / total_tokens
            perplexity = math.exp(avg_loss)
            results.append({'context_length': context_len, 'perplexity': perplexity})
            print(f"  Context {context_len}: PPL = {perplexity:.3f}")
        else:
            print(f"  Context {context_len}: No valid data")

    return results


def eval_passkey_retrieval(model, tokenizer, context_lengths=[16384, 32768], device='cuda', num_trials=5):
    """Evaluate passkey retrieval accuracy."""
    results = []
    for context_len in context_lengths:
        correct = 0
        for trial in range(num_trials):
            target_key = f"XXXXX{trial}YYYYY"
            filler_unit = "The apple is red. The sky is blue. The grass is green. "
            filler = filler_unit * (context_len // len(filler_unit) + 1)
            pos = random.randint(0, max(0, len(filler) - len(target_key) - 100))
            prompt_text = filler[:pos] + f"The pass key is {target_key}. Remember it. " + filler[pos:]
            prompt_text = prompt_text[:context_len - 50] + "\n\nWhat is the pass key? The pass key is"

            try:
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                response = tokenizer.decode(output_ids[inputs.input_ids.size(1):], skip_special_tokens=True).strip()
                if target_key in response:
                    correct += 1
            except RuntimeError:
                pass

        accuracy = correct / num_trials if num_trials > 0 else 0.0
        results.append({'context_length': context_len, 'accuracy': accuracy})
        print(f"  Context {context_len}: Accuracy = {accuracy:.2%}")

    return results


def eval_code_completion(model, tokenizer, device, num_examples=10):
    """Evaluate code completion BLEU score."""
    try:
        dataset = load_dataset("mbpp", split="test", cache_dir="/tmp/cache")
    except:
        print("  [Warning] MBPP dataset unavailable, skipping code evaluation")
        return []

    results = []
    chencherry = SmoothingFunction()
    for idx, example in enumerate(dataset.select(range(min(num_examples, len(dataset))))):
        code = example['code']
        split_point = len(code) // 2
        context = code[:split_point]
        target = code[split_point:]

        try:
            inputs = tokenizer(context, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=len(target) // 4 + 10, do_sample=False)
            prediction = tokenizer.decode(output_ids[inputs.input_ids.size(1):], skip_special_tokens=True).strip()
            reference = [target.split()]
            candidate = prediction.split()
            bleu = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
            results.append({'bleu': bleu})
        except RuntimeError:
            pass

    if results:
        avg_bleu = sum(r['bleu'] for r in results) / len(results)
        print(f"  Average BLEU: {avg_bleu:.4f}")
    return results


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmarks(
    model_name="Qwen/Qwen2.5-7B",
    device='cuda',
    context_lengths=[8192, 16384],  # Start small!
    output_dir="/tmp/benchmark_results"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    ).to(device)

    # ---- BASELINE EVALUATIONS ----
    print("\n=== BASELINE EVALUATION ===")
    print("Language Modeling...")
    lm_results_orig = eval_language_modeling(original_model, tokenizer, context_lengths, device)
    print("Passkey Retrieval...")
    passkey_results_orig = eval_passkey_retrieval(original_model, tokenizer, context_lengths, device)
    print("Code Completion...")
    code_results_orig = eval_code_completion(original_model, tokenizer, device)

    # ---- PATCH & CUSTOM EVALUATIONS ----
    print("\n=== PATCHING WITH CUSTOM ATTENTION ===")
    custom_model = patch_model_with_custom_attention(original_model, device)

    print("\n=== CUSTOM ATTENTION EVALUATION ===")
    print("Language Modeling...")
    lm_results_custom = eval_language_modeling(custom_model, tokenizer, context_lengths, device)
    print("Passkey Retrieval...")
    passkey_results_custom = eval_passkey_retrieval(custom_model, tokenizer, context_lengths, device)
    print("Code Completion...")
    code_results_custom = eval_code_completion(custom_model, tokenizer, device)

    # ---- SAVE RESULTS ----
    print("\n=== SAVING RESULTS ===")
    pd.DataFrame(lm_results_orig).to_csv(f"{output_dir}/lm_baseline.csv", index=False)
    pd.DataFrame(lm_results_custom).to_csv(f"{output_dir}/lm_adaptive.csv", index=False)
    pd.DataFrame(passkey_results_orig).to_csv(f"{output_dir}/passkey_baseline.csv", index=False)
    pd.DataFrame(passkey_results_custom).to_csv(f"{output_dir}/passkey_adaptive.csv", index=False)
    if code_results_orig:
        pd.DataFrame(code_results_orig).to_csv(f"{output_dir}/code_baseline.csv", index=False)
    if code_results_custom:
        pd.DataFrame(code_results_custom).to_csv(f"{output_dir}/code_adaptive.csv", index=False)

    print(f"\nResults saved to {output_dir}")
    print("\nFiles created:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    run_benchmarks(
        model_name="Qwen/Qwen2.5-7B",
        device='cuda',
        context_lengths=[8192, 16384],  # START SMALL!
        output_dir="/tmp/benchmark_results"
    )