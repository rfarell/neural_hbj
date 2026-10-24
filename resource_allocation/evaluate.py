"""Score every trained resource allocation run against the exact solution and draw the figures.

    python -m resource_allocation.evaluate

Writes results/metrics.csv (one row per run), results/summary.csv (mean and
standard deviation over seeds) and the figures in results/figures/.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import matplotlib.pyplot as plt

from neural_hjb import plotting, rollout
from neural_hjb.experiment import collect_runs, load_run

from .problem import ResourceAllocation
from .train import MODEL_DIR

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS = ["value_rel_err", "policy_mse", "policy_mse_off_switch", "reward_gap", "switch_time_err"]


def test_set(problem: ResourceAllocation, n: int = 4096, seed: int = 1234) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed test points, t uniform in [0, T] and x uniform in [x_min, x_max]."""
    g = torch.Generator().manual_seed(seed)
    t = problem.T * torch.rand(n, 1, generator=g)
    x = problem.sample_states(n, generator=g)
    return t, x


@torch.no_grad()
def closed_loop_paths(problem: ResourceAllocation, policy, x0: torch.Tensor, steps: int = 800):
    """Times [steps+1], states and controls [steps+1, B] of trajectories started at t = 0."""
    t0 = torch.zeros(x0.shape[0], 1)
    t, x, _ = rollout(problem, policy, t0, x0, steps=steps, num_points=steps + 1)
    u = torch.stack([policy(t_k, x_k) for t_k, x_k in zip(t, x)])
    return t[:, 0, 0], x[..., 0], u[..., 0]


def switch_times(t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """First time each control path drops below 1/2, linearly interpolated; T if it never does."""
    below = u < 0.5
    times = torch.full((u.shape[1],), t[-1].item())
    for b in range(u.shape[1]):
        idx = torch.nonzero(below[:, b])
        if len(idx) and idx[0, 0] > 0:
            k = idx[0, 0].item()
            frac = (u[k - 1, b] - 0.5) / (u[k - 1, b] - u[k, b])
            times[b] = t[k - 1] + frac * (t[k] - t[k - 1])
        elif len(idx):
            times[b] = t[0]
    return times


@torch.no_grad()
def evaluate(problem: ResourceAllocation, policy, value) -> Dict[str, float]:
    """Error metrics on the test set.

    value_rel_err          ||J - J*|| / ||J*||, i.e. the relative error of the reward-to-go
    policy_mse             mean (u - u*)^2
    policy_mse_off_switch  the same, excluding |t - t_s| < 0.1, where u* jumps
    reward_gap             relative reward lost by the learned policy, (E[V*] - E[R]) / E[V*],
                           with R the reward of simulating it from each test point
    switch_time_err        worst |switching time - t_s| over paths started at t = 0, x0 in [x_min, x_max]
    """
    t, x = test_set(problem)
    u, J = policy(t, x), value(t, x)
    u_star, J_star = problem.optimal_control(t, x), problem.optimal_cost(t, x)
    off_switch = ((t - problem.switching_time).abs() >= 0.1).squeeze(1)
    _, _, cost = rollout(problem, policy, t, x, steps=400)
    path_t, _, path_u = closed_loop_paths(problem, policy, torch.linspace(problem.x_min, problem.x_max, 9)[:, None])
    return {
        "value_rel_err": ((J - J_star).norm() / J_star.norm()).item(),
        "policy_mse": (u - u_star).pow(2).mean().item(),
        "policy_mse_off_switch": (u - u_star)[off_switch].pow(2).mean().item(),
        "reward_gap": ((cost.mean() - J_star.mean()) / -J_star.mean()).item(),
        "switch_time_err": (switch_times(path_t, path_u) - problem.switching_time).abs().max().item(),
    }


@torch.no_grad()
def plot_solution(problem: ResourceAllocation, policy, value, path: Path, reference=None) -> None:
    """Closed-loop control paths and the learned reward-to-go against the exact solution."""
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 4.2))

    x0 = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    t, _, u = closed_loop_paths(problem, policy, x0)
    u_star = problem.optimal_control(t[:, None], torch.ones(len(t), 1))[:, 0]
    ax_u.plot(t, u_star, color=plotting.INK, ls="--", lw=1.6, label="exact $u^*$")
    for i in range(len(x0)):
        ax_u.plot(t, u[:, i], color=plotting.SERIES[i], label=f"learned, $x_0$ = {x0[i, 0]:g}")
    ax_u.set_xlabel("Time t")
    ax_u.set_ylabel("Reinvested fraction u")
    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_title("Closed-loop control from t = 0", loc="left")
    ax_u.legend(loc="center left")

    t = torch.linspace(0, problem.T, 401)[:, None]
    ax_v.plot(t, problem.value_coefficient(t), color=plotting.INK, ls="--", lw=1.6, label=r"exact $\phi(t)$")
    for i, x_val in enumerate((1.0, 3.0, 5.0)):
        x = torch.full_like(t, x_val)
        ax_v.plot(t, -value(t, x) / x_val, color=plotting.SERIES[i], label=f"learned, x = {x_val:g}")
    if reference is not None:
        ref_value, ref_label = reference
        x = torch.full_like(t, 3.0)
        ax_v.plot(t, -ref_value(t, x) / 3.0, color=plotting.REFERENCE, lw=1.4, label=f"{ref_label}, x = 3")
    ax_v.set_xlabel("Time t")
    ax_v.set_ylabel("Reward-to-go per unit production, V(t, x) / x")
    ax_v.set_title(r"Reward-to-go: exact $V^* = x\,\phi(t)$", loc="left")
    ax_v.legend()
    for ax in (ax_u, ax_v):
        ax.axvline(problem.switching_time, color=plotting.GRID, lw=1.0, zorder=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.apply_style()
    runs = collect_runs(MODEL_DIR)
    if not runs:
        raise SystemExit(f"No checkpoints in {MODEL_DIR}; run `python -m resource_allocation.train` first.")
    rows = []
    for (variant, tag, seed), path in runs.items():
        problem, policy, value, ckpt = load_run(path, ResourceAllocation)
        print(f"evaluating {variant}/{tag} seed {seed}")
        rows.append({"variant": variant, "seed": seed, **evaluate(problem, policy, value),
                     "train_minutes": ckpt["train_seconds"] / 60})
    metrics = pd.DataFrame(rows).sort_values(["variant", "seed"])
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, float_format="%.6g")
    summary = metrics.groupby("variant")[METRICS + ["train_minutes"]].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(RESULTS_DIR / "summary.csv", index=False, float_format="%.4g")
    print(summary[[f"{m}_mean" for m in METRICS]].to_string(float_format="%.4g"))

    if ("full", "ra", 0) in runs:
        problem, policy, value, _ = load_run(runs[("full", "ra", 0)], ResourceAllocation)
        reference = None
        if ("original", "ra", 0) in runs:
            reference = (load_run(runs[("original", "ra", 0)], ResourceAllocation)[2], "original code")
        plot_solution(problem, policy, value, FIGURE_DIR / "solution.png", reference)


if __name__ == "__main__":
    main()
