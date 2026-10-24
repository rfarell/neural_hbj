"""Checkpointing and parallel training of (problem, variant, seed) runs."""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch

from .networks import PolicyNet, ValueNet
from .problem import ControlProblem
from .solver import VARIANTS, train_staged


def checkpoint_path(model_dir: Path, problem: ControlProblem, variant: str, seed: int) -> Path:
    return Path(model_dir) / variant / f"{problem.tag}_seed{seed}.pt"


def make_networks(
    problem: ControlProblem, hidden_dim: int = 64, value_scale: Optional[float] = None
) -> Tuple[PolicyNet, ValueNet]:
    """Fresh networks for a problem; value_scale defaults to problem.value_scale."""
    policy = PolicyNet(problem.state_dim, problem.control_dim, problem.constraint, hidden_dim)
    value = ValueNet(problem.state_dim, hidden_dim, problem.value_scale if value_scale is None else value_scale)
    return policy, value


def train_run(
    problem: ControlProblem,
    variant: str,
    seed: int,
    model_dir: Path,
    hidden_dim: int = 64,
    **train_kwargs,
) -> Path:
    """Train one run, saving the networks to a checkpoint and the loss history beside it as CSV."""
    torch.manual_seed(seed)
    policy, value = make_networks(problem, hidden_dim)
    start = time.time()
    history = train_staged(problem, policy, value, VARIANTS[variant], **train_kwargs)
    path = checkpoint_path(model_dir, problem, variant, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "problem": problem.config(),
            "variant": variant,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "value_scale": value.output_scale,
            "train": train_kwargs,
            "train_seconds": time.time() - start,
            "policy": policy.state_dict(),
            "value": value.state_dict(),
        },
        path,
    )
    pd.DataFrame(history).to_csv(path.with_suffix(".csv"), index_label="epoch")
    print(f"saved {path} ({time.time() - start:.0f}s)", flush=True)
    return path


def load_run(path: Path, make_problem: Callable[..., ControlProblem]) -> Tuple[ControlProblem, PolicyNet, ValueNet, dict]:
    """Rebuild the problem and networks stored in a checkpoint (networks in eval mode)."""
    ckpt = torch.load(path, weights_only=True)
    problem = make_problem(**ckpt["problem"])
    policy, value = make_networks(problem, ckpt["hidden_dim"], ckpt["value_scale"])
    policy.load_state_dict(ckpt["policy"])
    value.load_state_dict(ckpt["value"])
    policy.eval()
    value.eval()
    return problem, policy, value, ckpt


def _train_worker(args: Tuple) -> Path:
    problem, variant, seed, model_dir, kwargs = args
    torch.set_num_threads(1)
    return train_run(problem, variant, seed, model_dir, **kwargs)


def train_grid(
    problems: Iterable[ControlProblem],
    variants: Iterable[str],
    seeds: Iterable[int],
    model_dir: Path,
    jobs: int = 1,
    force: bool = False,
    **train_kwargs,
) -> List[Path]:
    """Train every (problem, variant, seed) combination, `jobs` single-threaded runs at a time.

    Combinations that already have a checkpoint are skipped unless `force` is set.
    """
    variants = list(variants)
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    tasks = [
        (problem, variant, seed, Path(model_dir), train_kwargs)
        for problem in problems
        for variant in variants
        for seed in seeds
    ]
    if not force:
        pending = []
        for problem, variant, seed, out_dir, kwargs in tasks:
            path = checkpoint_path(out_dir, problem, variant, seed)
            if path.exists():
                print(f"skipping {path}: already trained (use --force to retrain)")
            else:
                pending.append((problem, variant, seed, out_dir, kwargs))
        tasks = pending
    if not tasks:
        return []
    if jobs <= 1:
        return [train_run(problem, variant, seed, out_dir, **kwargs)
                for problem, variant, seed, out_dir, kwargs in tasks]
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        return list(pool.map(_train_worker, tasks))


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def collect_runs(model_dir: Path) -> Dict[Tuple[str, str, int], Path]:
    """Map (variant, tag, seed) to checkpoint path for every checkpoint under model_dir."""
    runs = {}
    for path in sorted(Path(model_dir).glob("*/*_seed*.pt")):
        tag, seed = path.stem.rsplit("_seed", 1)
        runs[(path.parent.name, tag, int(seed))] = path
    return runs
