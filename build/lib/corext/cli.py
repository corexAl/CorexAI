"""Command-line interface for COREX."""
import argparse
import sys
from pathlib import Path

from corext.config import get_corex_100m, get_corex_340m, get_corex_1b


def main():
    parser = argparse.ArgumentParser(
        prog="corex",
        description="COREX — Centralized Organized Reasoning & Extraction Matrix",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── train ─────────────────────────────────────
    tp = subparsers.add_parser("train", help="Train a COREX model")
    tp.add_argument("--preset", choices=["100m", "340m", "1b"], default="100m")
    tp.add_argument("--config", type=str, help="YAML config file path")
    tp.add_argument("--output-dir", type=str, default="checkpoints")
    tp.add_argument("--max-steps", type=int, help="Override max steps")
    tp.add_argument("--resume", type=str, help="Checkpoint to resume from")
    tp.add_argument("--lr", type=float, help="Override learning rate")
    tp.add_argument("--batch-size", type=int, help="Override batch size")

    # ── generate ──────────────────────────────────
    gp = subparsers.add_parser("generate", help="Generate text from a trained model")
    gp.add_argument("checkpoint", type=str)
    gp.add_argument("prompt", nargs="*", help="Prompt text (space-separated)")
    gp.add_argument("--max-tokens", type=int, default=128)
    gp.add_argument("--temperature", type=float, default=0.7)
    gp.add_argument("--top-p", type=float, default=0.9)
    gp.add_argument("--top-k", type=int, default=50)
    gp.add_argument("--repetition-penalty", type=float, default=1.1)
    gp.add_argument("--interactive", action="store_true")

    # ── info ──────────────────────────────────────
    ip = subparsers.add_parser("info", help="Show model architecture info")
    ip.add_argument("checkpoint", nargs="?", type=str, help="Checkpoint path (optional)")
    ip.add_argument("--preset", choices=["100m", "340m", "1b"])

    # ── bench ─────────────────────────────────────
    bp = subparsers.add_parser("bench", help="Benchmark model speed")
    bp.add_argument("checkpoint", type=str)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    from corext.training import train
    from corext.generation import load_model, generate, interactive_chat
    from corext.config import COREXConfig

    # ── Execute ───────────────────────────────────

    if args.command == "train":
        if args.config:
            cfg = COREXConfig.from_yaml(args.config)
        else:
            presets = {"100m": get_corex_100m, "340m": get_corex_340m, "1b": get_corex_1b}
            cfg = presets[args.preset]()

        if args.max_steps:
            cfg.max_steps = args.max_steps
        if args.lr:
            cfg.learning_rate = args.lr
        if args.batch_size:
            cfg.batch_size = args.batch_size

        train(config=cfg, output_dir=args.output_dir)

    elif args.command == "generate":
        model = load_model(args.checkpoint)

        if args.interactive:
            interactive_chat(model)
        else:
            prompt = " ".join(args.prompt) if args.prompt else "The future of AI is"
            result = generate(
                model, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            print("\nGenerated:")
            print("-" * 40)
            print(result)

    elif args.command == "info":
        if args.checkpoint:
            import torch
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            cfg = COREXConfig.from_dict(ckpt.get("config", {}))
        elif args.preset:
            presets = {"100m": get_corex_100m, "340m": get_corex_340m, "1b": get_corex_1b}
            cfg = presets[args.preset]()
        else:
            cfg = get_corex_100m()

        print(f"\nCOREX Configuration ({args.preset or 'custom'}):")
        for k, v in cfg.to_dict().items():
            print(f"  {k:>30s}: {v}")

    elif args.command == "bench":
        from corext.generation import benchmark_model
        model = load_model(args.checkpoint)
        results = benchmark_model(model)

        print("\nBenchmark Results:")
        for seq_len, m in results.items():
            print(f"  SeqLen {seq_len:>4d}: "
                  f"{m['avg_time_ms']:>8.1f}ms | "
                  f"{m['throughput_tokens_per_sec']:>10,.0f} tok/s")


if __name__ == "__main__":
    main()
