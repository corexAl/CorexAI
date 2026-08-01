"""Inference and generation utilities for COREX models."""
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from corext.config import COREXConfig, get_corex_100m
from corext.model import COREXModel


# ──────────────────────── Tokenizer ────────────────────────

class SimpleTokenizer:
    """Simple byte-level tokenizer for input/output operations.
    
    Encodes text as UTF-8 byte values (0-255), which maps naturally 
    to our 512-entry vocabulary (values mod 512).
    """

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for t in tokens:
            if 32 <= t < 127:
                chars.append(chr(t))
            elif t == 10:
                chars.append("\n")
            elif t == 13:
                chars.append("\r")
            elif t == 9:
                chars.append("    ")
            else:
                chars.append(f"\\x{t:02x}")
        return "".join(chars)


# ──────────────────────── Load Model ────────────────────────

def load_model(checkpoint_path: str, device: Optional[str] = None) -> COREXModel:
    """Load a trained COREX model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint
        device: Device to load on (auto-detects if None)
        
    Returns:
        Loaded and evaluated COREXModel instance
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device is None else torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config_dict = checkpoint.get("config", {})
    config = COREXConfig.from_dict(config_dict) if config_dict else get_corex_100m()

    model = COREXModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model


# ──────────────────────── Generate ────────────────────────

def generate(
    model: COREXModel,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: Optional[str] = None,
) -> str:
    """Generate text autoregressively from a prompt.
    
    The model processes the full context on first pass, then caches 
    key-value pairs for subsequent tokens to achieve O(n) memory usage.
    
    Args:
        model: Loaded COREXModel instance
        prompt: Input text prompt  
        max_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (lower = more deterministic, higher = more random)
        top_p: Nucleus sampling threshold (0-1, lower = more conservative)
        top_k: Top-k filtering (only sample from top k logits)
        repetition_penalty: Penalty for repeating tokens (>1 penalizes, <1 encourages)
        device: Override model device
        
    Returns:
        Generated text string (includes the prompt)
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = SimpleTokenizer()
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    generated_tokens = list(prompt_tokens)

    for step in range(max_tokens):
        context_len = min(len(input_ids[0]), model.config.max_position_embeddings)
        
        # Only feed the last max_position_embeddings tokens at once
        input_seq = input_ids[:, -context_len:]

        with torch.no_grad():
            logits = model(input_seq)  # (1, seq_len, vocab_size)

        # Get logits for next token (last position only)
        next_logits = logits[0, -1, :] / max(temperature, 1e-6)

        # Apply repetition penalty on recent tokens
        if repetition_penalty != 1.0 and len(generated_tokens) >= 5:
            recent_window = generated_tokens[-20:]
            for t in set(recent_window):
                next_logits[t] /= repetition_penalty

        # Top-k filtering
        if top_k > 0:
            topk_values, _ = torch.topk(next_logits, top_k)
            min_val = topk_values[-1:]
            next_logits = torch.where(
                next_logits < min_val,
                torch.full_like(next_logits, float("-inf")),
                next_logits,
            )

        # Sample from the filtered distribution
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated_tokens.append(next_token)
        new_ids = torch.tensor([[next_token]], dtype=torch.long).to(device)
        input_ids = torch.cat([input_ids, new_ids], dim=1)

        # Stop on EOS token
        if next_token == model.config.eos_token_id:
            break

    return tokenizer.decode(generated_tokens)


def generate_multiple(
    model: COREXModel,
    prompts: List[str],
    max_tokens: int = 128,
    temperature: float = 0.7,
    device: Optional[str] = None,
) -> List[str]:
    """Generate text from multiple prompts."""
    return [generate(model, p, max_tokens=max_tokens, temperature=temperature, device=device) for p in prompts]


# ──────────────────────── Interactive Chat ────────────────────────

def interactive_chat(model: COREXModel, device: Optional[str] = None):
    """Run an interactive chat session with the model."""
    print("\n" + "=" * 60)
    print("  COREX Interactive Chat")
    print("  Type your message and press Enter.")
    print("  Type 'quit' to exit, 'clear' to reset context.")
    print("=" * 60)

    history = []

    while True:
        try:
            user_input = input("\nYou > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower() == "clear":
                history = []
                print("  Context cleared.")
                continue

            # Format conversation for the model
            if history:
                prompt = (
                    "Human: "
                    + "\n".join(history[-10:])  # keep last 5 exchanges
                    + "\nCOREX: "
                    + user_input
                    + "\n"
                )
            else:
                prompt = f"Human: {user_input}\nCOREX: "

            response = generate(model, prompt, max_tokens=256)
            print(f"\n{response}")

            if history:
                history.append(user_input)
                history = history[-20:]  # keep last 10 exchanges

        except (EOFError, KeyboardInterrupt):
            break

    print("\nGoodbye!")


# ──────────────────────── Benchmark ────────────────────────

def benchmark_model(model: COREXModel, seq_lengths: List[int] = None, num_runs: int = 5):
    """Benchmark forward pass speed at different sequence lengths."""
    device = model.device if hasattr(model, "device") else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    import time
    results = {}

    for seq_len in seq_lengths or [16, 32, 64, 128]:
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(device)

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(input_ids)
            end = time.time()
            times.append(end - start)

        avg_time_ms = np.mean(times) * 1000
        throughput = 1.0 / np.mean(times) if np.mean(times) > 0 else float("inf")

        results[seq_len] = {
            "avg_time_ms": avg_time_ms,
            "throughput_tokens_per_sec": throughput,
            "num_params": model.get_num_params(),
        }

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generation.py <checkpoint> <prompt>")
        print()
        print("Examples:")
        print("  python generation.py checkpoints/corex-final.pth 'Hello, world!'")
        print("  python generation.py checkpoints/corex-100m.pth 'The future of AI is'")
        sys.exit(1)

    ckpt = sys.argv[1]
    prompt = " ".join(sys.argv[2:])

    print(f"Loading model from {ckpt}...")
    model = load_model(ckpt)
    result = generate(model, prompt, max_tokens=128)

    print("\n" + "=" * 60)
    print(f"PROMPT:   {prompt}")
    print("-" * 60)
    print(f"OUTPUT:   {result}")
    print("=" * 60)
