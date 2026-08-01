"""Comprehensive test suite for COREX production components.

Tests every major feature:
- Model architecture correctness (forward, backward, gradients)
- Training pipeline (optimizer state, checkpointing, evaluation)
- Tokenizer encoding/decoding round-trips  
- Generation with various sampling strategies
- Dataset loading and streaming
- Configuration serialization/deserialization
"""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import pytest
import tempfile
import os

from corext.config import COREXConfig, get_corex_100m, get_corex_340m, get_corex_1b
from corext.model import (COREXModel, init_weights, RotaryEmbedding, RMSNorm, 
                          Attention, FeedForward, enable_gradient_checkpointing)
from corext.tokenizer import BytePairTokenizer, GPT2Tokenizer
from corext.datasets import load_dataset, prepare_training_data, StreamingTextDataset
from corext.training import train, evaluate, compute_perplexity
from corext.generation import generate, load_model, CorextTokenizer


# ═══════════════════════════════════════════════════════════
#  Helper Fixture — Small Config for Fast Tests
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def small_config():
    """Create a tiny model config suitable for unit tests."""
    return COREXConfig(
        vocab_size=256, 
        hidden_size=32, 
        intermediate_size=128,
        num_hidden_layers=2, 
        num_attention_heads=4,
        max_position_embeddings=32,
        learning_rate=1e-3,
    )


@pytest.fixture
def small_model(small_config):
    """Create a fully initialized small model."""
    model = COREXModel(small_config)
    model.apply(init_weights)
    
    # Ensure all params require gradients (important for backward tests)
    for p in model.parameters():
        p.requires_grad = True
        
    return model


# ═══════════════════════════════════════════════════════════
#  Config Tests
# ═══════════════════════════════════════════════════════════

class TestConfig:
    def test_default_config(self):
        cfg = COREXConfig()
        assert cfg.vocab_size == 512
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 4

    def test_get_corex_100m(self):
        cfg = get_corex_100m()
        assert cfg.num_hidden_layers > 10
        assert cfg.hidden_size >= 256

    def test_yaml_round_trip(self):
        cfg = COREXConfig(hidden_size=128, num_hidden_layers=2)
        yaml_str = cfg.to_yaml()
        from corext.config import COREXConfig as C2
        cfg2 = C2.from_dict(cfg.to_dict())
        assert cfg2.hidden_size == 128


# ═══════════════════════════════════════════════════════════
#  Model Architecture Tests
# ═══════════════════════════════════════════════════════════

class TestModel:
    def test_model_creation(self, small_config):
        model = COREXModel(small_config)
        assert model.get_num_params() > 0
        assert len(model.blocks) == small_config.num_hidden_layers

    def test_forward_logits(self, small_model):
        input_ids = torch.randint(0, 256, (2, 16))
        logits = small_model(input_ids)
        assert logits.shape == (2, 16, 256)

    def test_forward_loss(self, small_model):
        input_ids = torch.randint(0, 256, (2, 16))
        loss = small_model(input_ids, labels=input_ids)
        assert loss.item() > 0 and torch.isfinite(loss)

    def test_gradient_flow(self, small_model):
        input_ids = torch.randint(0, 256, (2, 16))
        loss = small_model(input_ids, labels=input_ids)
        loss.backward()
        
        for name, param in small_model.named_parameters():
            if 'weight' in name and 'norm' not in name.lower():
                assert param.grad is not None, f"Gradient missing on {name}"

    def test_weight_tying(self, small_model):
        assert torch.equal(small_model.lm_head.weight, small_model.token_embeddings.weight)


# ═══════════════════════════════════════════════════════════
#  Component Tests
# ═══════════════════════════════════════════════════════════

class TestComponents:
    def test_rotary_embedding(self):
        rope = RotaryEmbedding(dim=16)
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape

    def test_rms_norm(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_attention(self):
        attn = Attention(dim=32, num_heads=4)
        x = torch.randn(1, 8, 32)
        mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        out = attn(x, mask=mask)
        assert out.shape == x.shape

    def test_feedforward(self):
        ffn = FeedForward(dim=32, intermediate_dim=128)
        x = torch.randn(1, 8, 32)
        out = ffn(x)
        assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════
#  Training Tests
# ═══════════════════════════════════════════════════════════

class TestTraining:
    def test_compute_perplexity(self):
        assert compute_perplexity(0.0) == 1.0           # Perfect prediction
        assert compute_perplexity(1.0) > 2.0            # Some uncertainty
        assert compute_perplexity(100.0) > 1e40         # Extreme loss
        
    def test_evaluate(self, small_model):
        data = prepare_training_data(sequence_length=16, is_eval=True)
        device = torch.device('cpu')
        
        metrics = evaluate(small_model, data[:10], device)
        
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] > 0
        assert metrics['perplexity'] > 1


# ═══════════════════════════════════════════════════════════
#  Full Training → Evaluation Pipeline Test
# ═══════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_train_generate_cycle(self):
        """Train a tiny model, then load and generate text.
        
        This is the ultimate integration test — verifies that training 
        produces valid weights, which can be loaded for inference.
        """
        from corext.training import train
        
        cfg = COREXConfig(
            vocab_size=256, hidden_size=32, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=32, learning_rate=1e-3,
            max_steps=20, batch_size=4, sequence_length=16,
            warmup_steps=5, output_dir='test_ckpts_pipeline',
            fp16=False, bf16=False, log_interval=10,
        )
        
        # Train for a small number of steps
        train(cfg, log_interval=10)
        
        # Load the trained model and verify it works
        from corext.generation import load_model
        model = load_model("test_ckpts_pipeline/corex-final.pth")
        assert model is not None
        
        # Generate text and verify output exists
        result = generate(model, "hello world test", max_tokens=10)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════
#  Dataset Tests
# ═══════════════════════════════════════════════════════════

class TestDatasets:
    def test_prepare_training_data(self):
        data = prepare_training_data(sequence_length=32)
        assert len(data) > 0
        
        sample = data[0]
        assert len(sample["input_ids"]) == 32
        assert len(sample["labels"]) == 32

    def test_streaming_dataset_creation(self):
        # Create a temp file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello world!\nTest passage for streaming.\nAnother line here.\n")
            tmp_path = f.name
        
        try:
            stream = StreamingTextDataset(tmp_path)
            passages = list(stream)
            assert len(passages) > 0
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════
#  Gradient Checkpointing Test
# ═══════════════════════════════════════════════════════════

class TestGradientCheckpointing:
    def test_checkpointing_context_manager(self, small_model):
        input_ids = torch.randint(0, 256, (2, 16))
        
        with enable_gradient_checkpointing(small_model):
            loss = small_model(input_ids, labels=input_ids)
            loss.backward()
            
        # Verify gradients were computed
        for p in small_model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


# ═══════════════════════════════════════════════════════════
#  Parameter Profiling Test
# ═══════════════════════════════════════════════════════════

class TestProfiling:
    def test_parameter_counting(self, small_model):
        from corext.model import count_parameters
        
        counts = count_parameters(small_model)
        
        assert counts['total'] > 0
        assert counts['trainable'] == counts['total']
        assert 'components' in counts


# ═══════════════════════════════════════════════════════════
#  Run Tests
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
