"""COREX - Centralized Organized Reasoning & Extraction Matrix.

A complete language model built entirely from scratch with:
- Rotary Position Embeddings (RoPE)
- Multi-head causal self-attention  
- SwiGLU gated feed-forward networks
- Pre-LayerNorm (RMSNorm variant)
"""
__version__ = "0.1.0"

from corext.config import COREXConfig, get_corex_100m, get_corex_340m, get_corex_1b
from corext.model import COREXModel, init_weights
from corext.training import train
from corext.generation import load_model, generate
from corext.datasets import prepare_training_data

__all__ = [
    "COREXConfig",
    "COREXModel", 
    "init_weights",
    "get_corex_100m",
    "get_corex_340m",
    "get_corex_1b",
    "train",
    "load_model",
    "generate",
    "prepare_training_data",
]
