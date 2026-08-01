"""Configuration for COREX models."""
from dataclasses import dataclass, field, asdict
import yaml
from typing import Optional


@dataclass
class COREXConfig:
    """Configuration for the COREX transformer model."""
    
    # Architecture
    vocab_size: int = 512
    hidden_size: int = 256
    intermediate_size: int = 768  
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    
    # Tokenization
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    sequence_length: int = 64
    gradient_accumulation_steps: int = 1
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Regularization
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    
    # Positional encoding
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[str] = None
    
    # Norm
    layer_norm_eps: float = 1e-5
    norm_type: str = "rms"  # rms or layer
    
    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 128
    do_sample: bool = True
    
    # Misc
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    output_dir: str = "outputs"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'COREXConfig':
        # Map some known aliases
        if 'intermediate_size' not in d and 'ffn_dim' in d:
            d['intermediate_size'] = d.pop('ffn_dim')
        if 'hidden_size' not in d and 'dim' in d:
            d['hidden_size'] = d.pop('dim')
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, path: str) -> 'COREXConfig':
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_yaml())


# ── Predefined model presets ──

def get_corex_100m() -> COREXConfig:
    """~100M parameter model."""
    hidden = 512
    return COREXConfig(
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=hidden * 3,
        num_hidden_layers=24,
        num_attention_heads=8,
        max_position_embeddings=2048,
        learning_rate=3e-4,
        warmup_steps=2000,
        max_steps=100000,
        batch_size=64,
        sequence_length=512,
        gradient_accumulation_steps=2,
    )


def get_corex_340m() -> COREXConfig:
    """~340M parameter model."""
    hidden = 768
    return COREXConfig(
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=hidden * 3,
        num_hidden_layers=32,
        num_attention_heads=12,
        max_position_embeddings=4096,
        learning_rate=2.5e-4,
        warmup_steps=5000,
        max_steps=200000,
        batch_size=64,
        sequence_length=1024,
        gradient_accumulation_steps=4,
    )


def get_corex_1b() -> COREXConfig:
    """~1B parameter model."""
    hidden = 1024
    return COREXConfig(
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=hidden * 3,
        num_hidden_layers=48,
        num_attention_heads=16,
        max_position_embeddings=4096,
        learning_rate=2e-4,
        warmup_steps=10000,
        max_steps=500000,
        batch_size=128,
        sequence_length=2048,
        gradient_accumulation_steps=4,
    )


if __name__ == '__main__':
    cfg = get_corex_100m()
    print(f"Model: COREX-100M")
    print(f"Vocab: {cfg.vocab_size}")
    print(f"Layers: {cfg.num_hidden_layers}")
    print(f"Hidden: {cfg.hidden_size}")
    print(f"Heads: {cfg.num_attention_heads}")
    print()
    print(cfg.to_yaml())
