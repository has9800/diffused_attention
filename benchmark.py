import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Iterable, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache
from datasets import load_dataset
import math
import random
import string
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import nltk

# Ensure NLTK punkt is downloaded (run once)
nltk.download('punkt', quiet=True)

# Your AdaptivePerHeadAttention class
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

# Custom Qwen2Attention
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
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = self.custom_softmax(attn_weights)

        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# Patch function
def patch_model_with_custom_attention(model, device):
    dtype = model.dtype
    for i, layer in enumerate(model.model.layers):
        custom_attn = CustomQwen2Attention(model.config, layer_idx=i)
        custom_attn.to(dtype=dtype, device=device)
        custom_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
        layer.self_attn = custom_attn
    return model

# Updated Evaluation functions
def eval_language_modeling(model, tokenizer, context_lengths, device, batch_size=1, max_model_len=131072, stride=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir="/data/cache")
    results = []

    for context_len in context_lengths:
        effective_context = min(context_len, max_model_len)
        total_nll = 0.0
        total_tokens = 0
        loss_fct = nn.CrossEntropyLoss(reduction='sum')  # Sum for accumulation

        for text in dataset["text"]:
            if not text.strip():  # Skip empty
                continue
            encodings = tokenizer(text, return_tensors="pt").to(device)
            seq_len = encodings.input_ids.size(1)
            if seq_len < 2:  # Need at least 2 tokens for shift
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
                total_tokens += shift_labels.numel()  # Count processed tokens

            if nlls:
                doc_nll = torch.stack(nlls).sum().item()
                total_nll += doc_nll

        if total_tokens > 0:
            avg_loss = total_nll / total_tokens
            perplexity = math.exp(avg_loss)
            results.append({
                'context_length': context_len,
                'perplexity': perplexity,
            })
        else:
            print(f"No valid data for context_len {context_len}")

    return results

def eval_passkey_retrieval(model, tokenizer, context_lengths=[32768, 65536], device='cuda'):
    results = []
    for context_len in context_lengths:
        correct = 0
        total = 10  # 10 trials
        for trial in range(total):
            target_key = f"XXXXX{trial}YYYYY"
            # Generate repetitive filler text
            filler_unit = "The apple is red. The sky is blue. The grass is green. "
            filler = filler_unit * (context_len // len(filler_unit) + 1)
            # Random position (depth)
            pos = random.randint(0, len(filler) - len(target_key) - 100)
            prompt_text = filler[:pos] + f"The pass key is {target_key}. Remember it. " + filler[pos:]
            prompt_text = prompt_text[:context_len - 50] + "\n\nWhat is the pass key? The pass key is"
            # Tokenize
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            # Generate
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            response = tokenizer.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True).strip()
            if target_key in response:
                correct += 1
        accuracy = correct / total
        results.append({
            'context_length': context_len,
            'accuracy': accuracy,
        })
    return results

def eval_code_completion(model, tokenizer, device, num_examples=50):
    full_dataset = load_dataset("mbpp", split="test", cache_dir="/data/cache")
    dataset = full_dataset.select(range(num_examples))  # Select subset as Dataset
    results = []
    chencherry = SmoothingFunction()
    for example in dataset:
        code = example['code']
        split_point = len(code) // 2
        context = code[:split_point]
        target = code[split_point:]
        # Tokenize context
        inputs = tokenizer(context, return_tensors="pt").to(device)
        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=len(target) // 4 + 10, do_sample=False)  # Approx tokens
        prediction = tokenizer.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True).strip()
        # BLEU with smoothing
        reference = [target.split()]
        candidate = prediction.split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
        results.append({
            'bleu': bleu,
            'language': 'python',
        })
    return results

# Main function to run benchmarks
def run_benchmarks(model_name="Qwen/Qwen2.5-7B", device='cuda', context_lengths=[8192, 16384, 32768, 65536]):
    os.makedirs("/data/cache", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data/cache")
    original_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, cache_dir="/data/cache").to(device)
    
    # Original evaluations
    lm_results_orig = eval_language_modeling(original_model, tokenizer, context_lengths, device)
    passkey_results_orig = eval_passkey_retrieval(original_model, tokenizer, context_lengths, device)
    code_results_orig = eval_code_completion(original_model, tokenizer, device)
    
    # Patch model
    custom_model = patch_model_with_custom_attention(original_model, device)
    
    # Custom evaluations
    lm_results_custom = eval_language_modeling(custom_model, tokenizer, context_lengths, device)
    passkey_results_custom = eval_passkey_retrieval(custom_model, tokenizer, context_lengths, device)
    code_results_custom = eval_code_completion(custom_model, tokenizer, device)
    
    # Save to CSV
    pd.DataFrame(lm_results_orig).to_csv("/data/lm_results_original.csv", index=False)
    pd.DataFrame(lm_results_custom).to_csv("/data/lm_results_custom.csv", index=False)
    pd.DataFrame(passkey_results_orig).to_csv("/data/passkey_results_original.csv", index=False)
    pd.DataFrame(passkey_results_custom).to_csv("/data/passkey_results_custom.csv", index=False)
    pd.DataFrame(code_results_orig).to_csv("/data/code_results_original.csv", index=False)
    pd.DataFrame(code_results_custom).to_csv("/data/code_results_custom.csv", index=False)
    
    # Plots
    # LM Perplexity
    plt.figure()
    plt.plot([r['context_length'] for r in lm_results_orig], [r['perplexity'] for r in lm_results_orig], label='Original')
    plt.plot([r['context_length'] for r in lm_results_custom], [r['perplexity'] for r in lm_results_custom], label='Custom')
    plt.xlabel('Context Length')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig("/data/plot_lm.png")
    
    # Passkey Accuracy
    plt.figure()
    plt.plot([r['context_length'] for r in passkey_results_orig], [r['accuracy'] for r in passkey_results_orig], label='Original')
    plt.plot([r['context_length'] for r in passkey_results_custom], [r['accuracy'] for r in passkey_results_custom], label='Custom')
    plt.xlabel('Context Length')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("/data/plot_passkey.png")
    
    # Code BLEU (average)
    avg_bleu_orig = sum(r['bleu'] for r in code_results_orig) / len(code_results_orig) if code_results_orig else 0
    avg_bleu_custom = sum(r['bleu'] for r in code_results_custom) / len(code_results_custom) if code_results_custom else 0
    print(f"Average BLEU Original: {avg_bleu_orig}")
    print(f"Average BLEU Custom: {avg_bleu_custom}")
    
    # Simple bar plot for BLEU
    plt.figure()
    plt.bar(['Original', 'Custom'], [avg_bleu_orig, avg_bleu_custom])
    plt.ylabel('Average BLEU')
    plt.savefig("/data/plot_code_bleu.png")

# Run
if __name__ == "__main__":
    run_benchmarks()