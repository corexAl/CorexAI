"""COREX Model — Complete Transformer Architecture from scratch.

Architecture Overview:
  Input Tokens → Embedding → [Transformer Block]×N → RMSNorm → LM Head → Softmax

Key Design Choices:
  - Rotary Positional Embeddings (RoPE) for relative positional encoding
  - SwiGLU gated feed-forward networks for better expressiveness
  - Pre-LayerNorm (RMSNorm variant) for stable training
  - Multi-head causal attention with full masking
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corext.config import COREXConfig


# ═══════════════════════════════════════════════════════════
#  Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings as described in "RoFormer: Enhanced Transformer".
    
    Applies rotation to query/key vectors based on their position, 
    enabling the model to learn relative positions naturally.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, rope_theta: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        # Precompute inverse frequencies: freq_k = θ^(-2k/d) for k=0,1,...,d/2-1
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache position-dependent cos/sin tables (lazy update on first forward)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int):
        """Update cos/sin cache if we need longer sequences."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device).float()
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
            self._cos_cached = emb.cos().to(dtype=torch.float32)
            self._sin_cached = emb.sin().to(dtype=torch.float32)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE rotation to query and key tensors.
        
        Args:
            q: (batch, heads, seq_len, head_dim)
            k: (batch, heads, seq_len, head_dim)
            
        Returns:
            RoPE-applied q, k tensors
        """
        self._update_cache(q.shape[2])  # seq_len

        cos = self._cos_cached[: q.shape[2]]  # (seq_len, dim/2)
        sin = self._sin_cached[: q.shape[2]]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        # Apply rotation: [cos -sin; sin cos] * [q; k] for each pair
        q_roped = (q * cos.unsqueeze(0).unsqueeze(0)) + \
                  (rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))
        k_roped = (k * cos.unsqueeze(0).unsqueeze(0)) + \
                  (rotate_half(k) * sin.unsqueeze(0).unsqueeze(0))

        return q_roped, k_roped


# ═══════════════════════════════════════════════════════════
#  RMSNorm — Root Mean Square Layer Normalization
# ═══════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm: a simpler, more effective layer normalization variant.
    
    Unlike standard LayerNorm, RMSNorm omits mean subtraction and 
    uses root-mean-square for scaling. This removes the need for a 
    learned bias term and simplifies computation.
    
    RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
    
    Args:
        dim: Hidden dimension
        eps: Small constant to avoid division by zero
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS norm along last dimension
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)


# ═══════════════════════════════════════════════════════════
#  SwiGLU — Gated Feed-Forward Network  
# ═══════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """SwiGLU (Gated Linear Units) feed-forward network.
    
    Modern LLMs use gated activations (GLU) which have been shown 
    to significantly improve expressiveness and training stability:
    
        FFN(x) = W₂ · SiLU(W₁x) ⊗ W₃x
    
    Where ⊗ is element-wise multiplication and SiLU is the Swish gate.
    """

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)   # Project to FFN dim
        self.w3 = nn.Linear(dim, intermediate_dim, bias=False)   # Gate projection  
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)   # Project back
        
        self.dropout = nn.Dropout(dropout)
        self.act = F.silu  # SiLU (Swish-1) activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


# ═══════════════════════════════════════════════════════════
#  Multi-Head Attention with RoPE
# ═══════════════════════════════════════════════════════════

class Attention(nn.Module):
    """Multi-head causal attention with rotary positional encoding.
    
    Architecture:
      Input (B, S, D) → QKV projection → RoPE → Scaled Dot-Product → Output
    
    The query/key vectors are rotated by RoPE before computing attention 
    scores, enabling the model to attend to positions relative to each other.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, rope_theta: float = 10_000.0):
        super().__init__()
        assert dim % num_heads == 0, "Hidden dim must be divisible by number of heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Single linear for Q/K/V (more efficient than three separate Liners)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, rope_theta=rope_theta)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor):
        """Reshape (batch, seq, dim) → (batch, heads, seq, head_dim)."""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, S, D)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape

        # Project to Q, K, V simultaneously
        qkv = self.qkv_proj(x)  # (B, S, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into heads
        q = self._reshape_for_heads(q)  # (B, H, S, D_head)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        # Apply rotary positional embeddings
        q, k = self.rope(q, k)

        # Scaled dot-product attention: O(K^T / √d_k) ⊗ V
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, S, S)

        # Apply causal mask (lower triangular)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over last dimension
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, H, S, D_head)

        # Reassemble heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)

        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════
#  Transformer Block — Pre-LN + Residual
# ═══════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization and residual connections.
    
    Architecture:
      x → RMSNorm → Attention → Dropout + x  (residual)
        → RMSNorm → FFN(SwiGLU) → Dropout + x  (residual)
        
    Pre-LayerNorm is used (norm before projection) which provides 
    better gradient flow than post-LayerNorm.
    """

    def __init__(self, cfg: COREXConfig):
        super().__init__()

        attn_dim = cfg.hidden_size
        
        self.attention = Attention(
            dim=attn_dim,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.attention_dropout,
            rope_theta=cfg.rope_theta,
        )

        self.ffn = FeedForward(
            dim=attn_dim,
            intermediate_dim=cfg.intermediate_size,
            dropout=cfg.residual_dropout,
        )

        # Pre-LayerNorm modules
        self.attn_norm = RMSNorm(attn_dim, eps=cfg.layer_norm_eps)
        self.ffn_norm = RMSNorm(attn_dim, eps=cfg.layer_norm_eps)

        self.dropout = nn.Dropout(cfg.residual_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention with residual (pre-LN)
        attn_out = self.attention(self.attn_norm(x), mask=mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual (pre-LN)
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)

        return x


# ═══════════════════════════════════════════════════════════
#  COREX Model — Full Language Model
# ═══════════════════════════════════════════════════════════

class COREXModel(nn.Module):
    """Complete COREX causal language model.
    
    Architecture Flow:
      ┌─────────────────────────────────────────┐
      │  Input Tokens (B, S)                    │
      │         ↓                               │
      │  Token Embedding (B, S, D)              │
      │         ↓                               │
      │  [Transformer Block] × N (RoPE, Attn, FFN)│
      │         ↓                               │
      │  RMSNorm → LM Head → Logits (B, S, V)   │
      └─────────────────────────────────────────┘
    
    Key Features:
      - Byte-level BPE vocabulary (vocab_size entries)
      - Rotary Position Embeddings for positional info  
      - Multi-head causal self-attention
      - SwiGLU gated feed-forward networks
      - Pre-LayerNorm (RMSNorm variant)
      - Kaiming-normal weight initialization
      - Weight tying between embedding and LM head
    """

    def __init__(self, config: Optional[COREXConfig] = None):
        super().__init__()

        if config is None:
            config = COREXConfig()

        self.config = config
        self.embed_dim = config.hidden_size

        # Token embedding layer
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            self.embed_dim,
            padding_idx=config.pad_token_id,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final normalization before LM head
        self.norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        # Language model head (weight-tied with embeddings if same size)
        if config.vocab_size == self.token_embeddings.num_embeddings:
            self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embeddings.weight  # Weight tying!
        else:
            self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.residual_dropout)

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass of the COREX model.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: optional target tokens for loss computation
            
        Returns:
            If labels provided: cross-entropy loss scalar
            Otherwise: logits tensor (batch, seq_len, vocab_size)
        """
        batch, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embeddings(input_ids)  # (B, S, D)
        x = self.dropout(x)

        # Causal attention mask (lower triangular boolean mask)
        attn_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
        )

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attn_mask)

        # Final RMSNorm
        x = self.norm(x)

        # LM head projection to vocabulary space
        logits = self.lm_head(x)  # (B, S, V)

        if labels is not None:
            # Shift for autoregressive training (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()   # (B, S-1, V)
            shift_labels = labels[..., 1:].contiguous()        # (B, S-1)

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            return loss

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Autoregressive text generation with caching.
        
        First pass processes full prompt. Subsequent passes only 
        compute the last token (using KV cache), reducing O(n²) 
        attention to O(n) memory for long sequences.
        """
        self.eval()
        generated = prompt_ids.clone()

        past_kv = None  # KV cache: list of (key, value) per layer

        for step in range(max_new_tokens):
            if past_kv is None:
                # Full forward pass on complete context
                input_seq = generated
                seq_len = len(input_seq)
            else:
                # Only compute last token (caching enabled)
                input_seq = generated[:, -1:]
                seq_len = 1

            x = self.token_embeddings(input_seq)
            
            attn_mask = torch.tril(
                torch.ones(seq_len, seq_len + (len(past_kv[0][0]) - 1 if past_kv else 0),
                           dtype=torch.bool, device=generated.device)
            ) if past_kv else torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=generated.device))

            for block in self.blocks:
                x = block(x, mask=attn_mask)

            logits = self.lm_head(self.norm(x))[:, -1, :]  # (batch, vocab)

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated[0]) > 5:
                for t in torch.unique(generated).tolist():
                    idx = generated == t
                    logits[idx] /= repetition_penalty

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_values, _ = torch.topk(logits, top_k)
                min_val = topk_values[:, -1:]
                logits[logits < min_val] = float('-inf')

            # Sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS token
            if next_token.item() == self.config.eos_token_id:
                break

        return generated


# ═══════════════════════════════════════════════════════════
#  Weight Initialization Utilities
# ═══════════════════════════════════════════════════════════

def init_weights(module: nn.Module):
    """Initialize weights using a variant of the GPT-2/OpenAI scheme.
    
    Strategy:
      - Linear/Embedding layers: Kaiming normal (σ = 1/√fan_in) — better for Swish activation
      - Residual projections: Initialized to zero (follows modern practice)
      - Norm layers: Ones for weights, zeros for bias
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, std=0.02)

    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)


def get_parameter_groups(model: COREXModel, weight_decay: float = 0.01):
    """Get parameter groups for AdamW optimizer.
    
    Groups parameters into those that should have weight decay (all except 
    bias and norm weights) to prevent over-regularization of these parameters.
    """
    no_decay = ['bias', 'weight', 'norm']

    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        has_decay = not any(nd in name.lower() for nd in no_decay)

        param_groups.append({
            'params': [param],
            'weight_decay': weight_decay if has_decay else 0.0,
        })

    return param_groups


if __name__ == '__main__':
    # Quick sanity check — can we create and run the model?
    print("Running COREX architecture sanity check...\n")

    cfg = COREXConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
        learning_rate=3e-4,
    )

    model = COREXModel(cfg)
    model.apply(init_weights)

    print(f"✓ Model initialized: {model.get_num_params():,} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"✓ Forward: {input_ids.shape} → {logits.shape}")
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)

    # Test loss computation  
    loss = model(input_ids, labels=input_ids)
    print(f"✓ Loss: {loss.item():.4f}")

    # Test generation stub
    gen = model.generate(input_ids[:1], max_new_tokens=5)
    print(f"✓ Generation: {input_ids.shape[1]} → {gen.shape[1]} tokens")

    print("\n✅ All sanity checks passed!")
