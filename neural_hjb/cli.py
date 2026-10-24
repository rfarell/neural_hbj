"""Command-line options shared by the training scripts."""

import argparse

from .experiment import default_jobs
from .solver import VARIANTS


def training_parser(description: str, rollout_steps: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--variants", nargs="+", default=["full"], choices=sorted(VARIANTS),
                        help="Loss variants to train; 'full' is the method, the others are ablations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds, one run each")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs per stage")
    parser.add_argument("--stages", type=int, default=5, help="Number of backward-in-time stages")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=rollout_steps, help="RK4 steps per training rollout")
    parser.add_argument("--jobs", type=int, default=default_jobs(), help="Runs to train in parallel")
    parser.add_argument("--force", action="store_true", help="Retrain runs that already have a checkpoint")
    return parser


def train_kwargs(args: argparse.Namespace) -> dict:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    return {
        "hidden_dim": args.hidden_dim,
        "stages": args.stages,
        "epochs_per_stage": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "rollout_steps": args.rollout_steps,
    }
