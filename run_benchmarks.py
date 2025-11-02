#!/usr/bin/env python3
"""
Benchmark Mistral attention variants on WikiText-103.

The script evaluates three configurations:
    - baseline: unmodified Mistral attention
    - fixed_two_pool: two-pool renormalisation with fixed epsilon/temperature
    - adaptive_per_head: layer-depth adaptive epsilon/temperature

Perplexities at the requested context lengths are written to a CSV file with
columns: context_length, method, perplexity.
"""

import argparse
import csv
import math
import os
from array import array
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from MonkeyPatchLM import patch_attention, unpatch_attention


def load_mistral_model(
    model_name: str,
    hf_token: Optional[str],
    dtype: str,
    device: str,
    rope_factor: float,
    quantization: str,
    max_context_length: int,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    quantization = quantization.lower()
    torch_dtype = getattr(torch, dtype)

    load_kwargs = {}
    if hf_token:
        load_kwargs["use_auth_token"] = hf_token

    if quantization in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for 4bit/8bit quantization. "
                "Install it with `pip install bitsandbytes`."
            ) from exc

        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch_dtype
    elif quantization == "none":
        load_kwargs["torch_dtype"] = torch_dtype
    else:
        raise ValueError("quantization must be one of: none, 8bit, 4bit")

    rope_kwargs = {}
    effective_rope_factor = rope_factor
    if rope_factor != 1.0:
        rope_kwargs["rope_scaling"] = {"type": "linear", "factor": rope_factor}

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs, **rope_kwargs)
    except TypeError as exc:
        if "rope_scaling" in str(exc):
            if rope_kwargs:
                print(
                    f"Warning: model {model_name} does not accept rope_scaling; "
                    "falling back to default context window."
                )
                effective_rope_factor = 1.0
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            raise

    if quantization == "none":
        target_device = torch.device(device)
        model.to(target_device)
    else:
        target_device = model.device

    model.eval()
    model.config.use_cache = False

    tok_kwargs = {}
    if hf_token:
        tok_kwargs["use_auth_token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(max_context_length, int(effective_rope_factor * 8192))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, str(target_device)


def build_token_buffer(
    tokenizer: AutoTokenizer,
    split: str,
    hf_cache_dir: Optional[str],
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, cache_dir=hf_cache_dir)
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define an EOS token.")

    token_buffer = array("I")

    for example in dataset:
        text = example["text"]
        if not text:
            continue
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not encoded:
            continue
        token_buffer.extend(encoded)
        token_buffer.append(eos_id)

    if not token_buffer:
        raise RuntimeError("Token buffer is empty. Check the dataset/tokenizer setup.")

    tokens = torch.tensor(token_buffer, dtype=torch.long)
    return tokens


def batch_iterator(
    token_buffer: torch.Tensor,
    context_length: int,
    batch_size: int,
) -> Iterable[torch.Tensor]:
    total_tokens = (token_buffer.numel() // context_length) * context_length
    trimmed = token_buffer[:total_tokens]
    if trimmed.numel() == 0:
        return

    sequences = trimmed.view(-1, context_length)
    num_sequences = sequences.size(0)

    for start in range(0, num_sequences, batch_size):
        end = start + batch_size
        if end > num_sequences:
            break
        yield sequences[start:end]


def evaluate_perplexity(
    model: AutoModelForCausalLM,
    token_buffer: torch.Tensor,
    context_length: int,
    batch_size: int,
    device: Union[str, torch.device],
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    target_device = torch.device(device) if isinstance(device, str) else device
    device_str = str(target_device)

    for batch_idx, batch_cpu in enumerate(batch_iterator(token_buffer, context_length, batch_size)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = batch_cpu.to(target_device)

        with torch.no_grad():
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss

        n_tokens = batch.size(0) * batch.size(1)
        total_loss += float(loss.item()) * n_tokens
        total_tokens += n_tokens

        del batch
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_tokens == 0:
        raise RuntimeError(f"No batches processed for context length {context_length}.")

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    return perplexity


def run_evaluation(
    model: AutoModelForCausalLM,
    token_buffer: torch.Tensor,
    context_lengths: Sequence[int],
    batch_size: int,
    device: Union[str, torch.device],
    max_batches: Optional[int],
    results_path: Path,
) -> None:
    methods = ["baseline", "fixed_two_pool", "adaptive_per_head"]
    rows = []

    for method in methods:
        if method == "baseline":
            unpatch_attention(model)
        elif method == "fixed_two_pool":
            patch_attention(model, mode="fixed_two_pool")
        elif method == "adaptive_per_head":
            patch_attention(model, mode="adaptive_per_head")

        for context_length in context_lengths:
            print(f"Evaluating {method} at context {context_length} ...", flush=True)
            try:
                ppl = evaluate_perplexity(
                    model=model,
                    token_buffer=token_buffer,
                    context_length=context_length,
                    batch_size=batch_size,
                    device=device,
                    max_batches=max_batches,
                )
                ppl_str = f"{ppl:.6f}"
            except RuntimeError as e:
                print(f"Warning: failed at context {context_length} for {method}: {e}")
                ppl_str = "nan"
            rows.append(
                {
                    "context_length": context_length,
                    "method": method,
                    "perplexity": ppl_str,
                }
            )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["context_length", "method", "perplexity"])
        writer.writeheader()
        writer.writerows(rows)

    unpatch_attention(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mistral attention variants on WikiText-103.")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--wikitext-split", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--rope-factor", type=float, default=2.0)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="4bit")
    parser.add_argument("--output", type=Path, default=Path("results/mistral_attention_benchmark.csv"))
    args = parser.parse_args()

    hf_token = os.environ.get(args.hf_token_env)
    if hf_token is None:
        print(f"Warning: ${args.hf_token_env} not set. Proceeding without an auth token.")

    max_context = max(args.context_lengths)

    model, tokenizer, model_device = load_mistral_model(
        model_name=args.model_name,
        hf_token=hf_token,
        dtype=args.dtype,
        device=args.device,
        rope_factor=args.rope_factor,
        quantization=args.quantization,
        max_context_length=max_context,
    )

    token_buffer = build_token_buffer(
        tokenizer=tokenizer,
        split=args.wikitext_split,
        hf_cache_dir=args.hf_cache_dir,
        max_samples=args.max_samples,
    )

    run_evaluation(
        model=model,
        token_buffer=token_buffer,
        context_lengths=args.context_lengths,
        batch_size=args.batch_size,
        device=model_device,
        max_batches=args.max_batches,
        results_path=args.output,
    )

    print(f"Benchmark complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
