"""Training pipeline for COREX — from data to checkpoint."""
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from corext.config import COREXConfig, get_corex_100m
from corext.model import COREXModel, init_weights
from corext.datasets import load_dataset, prepare_training_data


def train(
    config: COREXConfig = None,
    output_dir: str = "checkpoints",
    log_interval: int = 100,
    eval_interval: int = 1000,
    save_interval: int = 5000,
):
    """Full training loop for COREX.
    
    Implements:
      - AdamW optimizer with weight decay and bias/norm exclusion  
      - Linear warmup + cosine annealing LR schedule
      - Gradient accumulation for effective batch size control
      - Gradient clipping (max norm = 1.0)
      - Mixed precision (bf16/fp16) via torch.cuda.amp
      - Periodic evaluation and checkpointing
      
    Args:
        config: Model configuration (uses COREX-100m preset if None)
        output_dir: Directory to save checkpoints and logs
        log_interval: Training steps between console logging
        eval_interval: Steps between evaluation
        save_interval: Steps between saving checkpoints
        
    Returns:
        Dictionary of training history metrics
    """

    # ──────────────────────── Setup ────────────────────────
    if config is None:
        config = get_corex_100m()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  COREX Training — Starting")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ──────────────────────── Model Creation ────────────────────────
    model = COREXModel(config)
    model.apply(init_weights)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Architecture:")
    print(f"    Parameters:   {total_params:>12,}")
    print(f"    Hidden dim:   {config.hidden_size:>12}")
    print(f"    Layers:       {config.num_hidden_layers:>12}")
    print(f"    Heads:        {config.num_attention_heads:>12}")
    print(f"    Vocab:        {config.vocab_size:>12}")

    # ──────────────────────── Data Loading ────────────────────────
    print(f"\n  Data:")
    train_data = prepare_training_data(
        sequence_length=config.sequence_length, is_eval=False
    )
    eval_data = prepare_training_data(
        sequence_length=config.sequence_length, is_eval=True
    )
    print(f"    Train:      {len(train_data):>12,} sequences")
    print(f"    Eval:       {len(eval_data):>12,} sequences")

    # ──────────────────────── Optimizer & Scheduler ────────────────────────
    # Parameter groups: exclude bias and norm weights from weight decay
    param_groups = [
        {"params": [], "weight_decay": 0.0},   # No-decay params
        {"params": [], "weight_decay": config.weight_decay},  # Decay params
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        has_decay = not any(nd in name.lower() for nd in ["bias", "norm"])
        group_idx = 1 if has_decay else 0
        param_groups[group_idx]["params"].append(param)

    # Ensure each group has params (in case model is empty)
    for i, g in enumerate(param_groups):
        if not g["params"]:
            g["weight_decay"] = 0.0

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    # Linear warmup + cosine decay scheduler
    total_steps = config.max_steps
    warmup_steps = config.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision
    use_amp = False
    if config.bf16 and torch.cuda.is_available():
        use_amp = torch.cuda.is_bf16_supported()
    elif config.fp16:
        use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n  Training:")
    print(f"    Optimizer:    AdamW (lr={config.learning_rate:.2e})")
    print(f"    Weight decay: {config.weight_decay}")
    print(f"    Warmup:       {warmup_steps} steps")
    print(f"    Batch size:   {config.batch_size * config.gradient_accumulation_steps}")
    print(f"    AMP:          {'bf16' if use_amp else 'fp32'}")

    # ──────────────────────── Training Loop ────────────────────────
    model.train()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = []  # training history
    global_step = 0
    n_accumulated = 0
    running_loss = 0.0

    start_time = time.time()

    pbar = tqdm(total=total_steps, desc="Training", unit="steps")

    while global_step < total_steps:
        # Get batch (cyclic through dataset)
        data_idx = global_step % len(train_data)
        sample = train_data[data_idx]
        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        labels = torch.tensor([sample["labels"]]).to(device)

        # Forward pass with AMP if enabled
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(input_ids, labels=labels)

        # Normalize for gradient accumulation
        loss = loss / config.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()
        n_accumulated += 1

        if n_accumulated >= config.gradient_accumulation_steps:
            # Gradient clipping (max norm)
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            n_accumulated = 0

        # Tracking
        running_loss += loss.item() * config.gradient_accumulation_steps
        global_step += 1

        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            tokens_per_sec = (config.batch_size * config.sequence_length * 
                              log_interval) / elapsed if elapsed > 0 else 0

            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
            print(f"\n  Step {global_step:>8,} | Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | Tokens/s: {tokens_per_sec:,.0f}")

            running_loss = 0.0

        # Evaluation
        if eval_interval > 0 and global_step % eval_interval == 0 and global_step > 0:
            eval_loss = evaluate(model, eval_data, device)
            pbar.write(f"  └─ Eval Loss: {eval_loss:.4f}")
            metrics.append({"step": global_step, "loss": avg_loss, "eval_loss": eval_loss})

        # Save checkpoint
        if save_interval > 0 and global_step % save_interval == 0 and global_step > 0:
            save_checkpoint(
                model, optimizer, scheduler, config,
                checkpoint_path=str(output_dir / f"corex-{global_step}.pth"),
            )

        pbar.update(1)

    # ──────────────────────── Final Save ────────────────────────
    final_path = output_dir / "corex-final.pth"
    save_checkpoint(model, optimizer, scheduler, config, str(final_path))

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✓ Training Complete!")
    print(f"    Steps:          {global_step:>10,}")
    print(f"    Final Loss:     {metrics[-1]['loss'] if metrics else 'N/A':>10.4f}")
    print(f"    Total Time:     {elapsed/3600:>10.2f}h")
    print(f"    Checkpoint:     {final_path}")
    print(f"{'='*60}\n")

    return metrics


def evaluate(model: COREXModel, data, device) -> float:
    """Compute average loss on a dataset."""
    model.eval()
    total_loss = 0.0
    n = min(50, len(data))  # sample for speed

    with torch.no_grad():
        for i in range(n):
            batch = data[i]
            input_ids = torch.tensor([batch["input_ids"]]).to(device)
            labels = torch.tensor([batch["labels"]]).to(device)
            total_loss += model(input_ids, labels=labels).item()

    return total_loss / n


def save_checkpoint(model, optimizer, scheduler, config, checkpoint_path):
    """Save full training state to disk."""
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.to_dict(),
        "step": 0,
    }, checkpoint_path)


if __name__ == "__main__":
    # Quick test — should complete in ~30s on CPU with small config
    config = get_corex_100m()
    config.max_steps = 500   # tiny for testing
    config.sequence_length = 64
    config.batch_size = 8

    train(config, output_dir="test_checkpoints")
