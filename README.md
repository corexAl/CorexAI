<img width="2064" height="512" alt="corex_banner" src="https://github.com/user-attachments/assets/a58c0144-0087-4548-b091-006224c0e370" />

## COREX 

**COREX** (Centralized Organized Reasoning & Extraction Matrix) is a complete, self-contained language model built entirely from scratch with zero dependencies on HuggingFace or other AI frameworks. Just pure PyTorch transformer architecture working code.

## What's Inside

| Module | Purpose |
|---|---|
| `corext/config.py` | Model configuration presets (100M/340M/1B params) |
| `corext/model.py` | Full transformer: RoPE, Multi-head attention, SwiGLU FFN |
| `corext/training.py` | Training loop with AdamW, LR scheduling, gradient clipping |
| `corext/generation.py` | Autoregressive text generation with KV caching |
| `corext/datasets.py` | Dataset loading (C4/Wikitext/synthetic fallback) |
| `corext/cli.py` | Command-line interface for train/generate/info/bench |
| `corext/configs/` | YAML presets for each model size |

## Architecture Details

- **Tokenization**: Byte-level vocabulary with BPE merges (512 base tokens)
- **Positional Encoding**: Rotary Embeddings (RoPE) for relative position awareness
- **Attention**: Multi-head causal self-attention with RoPE on Q/K vectors
- **FFN**: SwiGLU gated feed-forward — `W₂ · SiLU(W₁x) ⊗ W₃x`
- **Normalization**: RMSNorm (pre-LayerNorm variant, simpler and effective)
- **Residual Scale**: Kaiming-normal init with proper scaling for deep networks

## Quick Start

### Install

```bash
pip install corext
```

### Train a Model

```bash
# Using preset
corex train --preset 100m --output-dir my_model/

# Or from YAML config  
corex train --config configs/example-100m.yaml

# Custom options
corex train --preset 340m --max-steps 50000 --lr 2e-4 --batch-size 64
```

### Generate Text

```bash
# Single prompt
corex generate --checkpoint checkpoints/corex-final.pth "Hello, world!"

# Interactive mode  
corex generate --checkpoint my_model/corex-final.pth --interactive
```

### Inspect Model

```bash
corex info --preset 100m
corex bench checkpoints/corex-final.pth
```

## Programmatic API

```python
from corext import COREXConfig, get_corex_100m, train, load_model, generate

# Training
cfg = get_corex_100m()
cfg.max_steps = 10000
history = train(cfg, output_dir="outputs/")

# Inference
model = load_model("outputs/corex-final.pth")
text = generate(model, "The future of AI is", max_tokens=50)
print(text)
```

## Model Zoo

| Model | Params | Layers | Hidden Dim | Attention Heads | Sequence Length |
|---|---|---|---|---|---|
| corex-100m | ~100M | 24 | 512 | 8 | 512 |
| corex-340m | ~340M | 32 | 768 | 12 | 1024 |
| corex-1b | ~1B | 48 | 1024 | 16 | 2048 |

## Training Configuration

All configs are in `corext/configs/` and support these hyperparameters:

- **Architecture**: vocab_size, hidden_size, intermediate_size, num_layers, num_heads
- **Training**: learning_rate, weight_decay, warmup_steps, max_steps
- **Batching**: batch_size, sequence_length, gradient_accumulation_steps
- **Regularization**: attention_dropout, residual_dropout
- **Optimization**: adam_beta1/beta2, adam_eps
- **Precision**: fp16, bf16 (auto-detected on GPU)

This project combines human engineering and AI-assisted code generation. All code included in this repository has been reviewed and integrated by the maintainer.
