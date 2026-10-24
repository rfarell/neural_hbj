"""Score every trained motion control run against the exact solution and draw the figures.

    python -m motion_control.evaluate

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

from .problem import MotionControl
from .train import MODEL_DIR

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS = ["value_rel_err", "policy_err", "cost_gap", "value_mse", "legacy_mse_u"]


def test_set(problem: MotionControl, n: int = 4096, seed: int = 1234) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed test points, t uniform in [0, T] and x uniform in the training box."""
    g = torch.Generator().manual_seed(seed)
    t = problem.T * torch.rand(n, 1, generator=g)
    x = problem.sample_states(n, generator=g)
    return t, x


@torch.no_grad()
def evaluate(problem: MotionControl, policy, value) -> Dict[str, float]:
    """Error metrics on the test set.

    value_rel_err  ||J - J*|| / ||J*||
    policy_err     mean ||u - u*||^2 where u* is unique (||x|| > T - t)
    cost_gap       relative excess cost of the learned policy, (E[C] - E[J*]) / E[J*],
                   with C the cost of simulating it from each test point
    value_mse      mean (J - J*)^2
    legacy_mse_u   mean over all points and components of (u - u*)^2, the original repo's metric
    """
    t, x = test_set(problem)
    u, J = policy(t, x), value(t, x)
    u_star, J_star = problem.optimal_control(t, x), problem.optimal_cost(t, x)
    unique = (x.norm(dim=1, keepdim=True) > problem.T - t).squeeze(1)
    _, _, cost = rollout(problem, policy, t, x, steps=200)
    return {
        "value_rel_err": ((J - J_star).norm() / J_star.norm()).item(),
        "policy_err": (u - u_star)[unique].pow(2).sum(dim=1).mean().item(),
        "cost_gap": ((cost.mean() - J_star.mean()) / J_star.mean()).item(),
        "value_mse": (J - J_star).pow(2).mean().item(),
        "legacy_mse_u": (u - u_star).pow(2).mean().item(),
    }


def plot_error_vs_dimension(metrics: pd.DataFrame, path: Path) -> None:
    panels = [
        ("value_rel_err", "Cost-to-go error", r"$\|J - J^*\| \,/\, \|J^*\|$"),
        ("policy_err", "Control error", r"mean $\|u - u^*\|^2$ where $u^*$ is unique"),
        ("cost_gap", "Excess cost of the learned policy", r"$(\mathbb{E}[C] - \mathbb{E}[J^*]) \,/\, \mathbb{E}[J^*]$"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8))
    for ax, (column, title, ylabel) in zip(axes, panels):
        for variant, style in plotting.VARIANT_STYLE.items():
            runs = metrics[metrics.variant == variant]
            if runs.empty:
                continue
            stats = runs.groupby("dim")[column].agg(["mean", "min", "max"])
            ax.errorbar(
                stats.index, stats["mean"],
                yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]],
                color=style["color"], marker=style["marker"], ls=style["ls"], label=style["label"], capsize=3,
            )
        ax.set_yscale("log")
        ax.set_xticks(sorted(metrics.dim.unique()))
        ax.set_xlabel("State dimension n")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def plot_slices(problem: MotionControl, policy, value, path: Path, times=(0.0, 0.5, 0.9), grid: int = 161) -> None:
    """2-D maps of the cost-to-go error and the control error at a few times."""
    w = problem.half_width
    axis = torch.linspace(-w, w, grid)
    X, Y = torch.meshgrid(axis, axis, indexing="xy")
    x = torch.stack((X.flatten(), Y.flatten()), dim=1)
    rows = []
    for time in times:
        t = torch.full((x.shape[0], 1), time)
        J_err = (value(t, x) - problem.optimal_cost(t, x)).view(grid, grid)
        u = policy(t, x)
        u_err = (u - problem.optimal_control(t, x)).norm(dim=1).view(grid, grid)
        rows.append((time, J_err, u_err, u))
    j_lim = max(r[1].abs().max().item() for r in rows)
    u_lim = max(r[2].max().item() for r in rows)

    fig, axes = plt.subplots(2, len(times), figsize=(4.0 * len(times), 7.4), constrained_layout=True)
    extent = [-w, w, -w, w]
    skip = slice(None, None, 16)
    for col, (time, J_err, u_err, u) in enumerate(rows):
        radius = problem.T - time
        im_j = axes[0, col].imshow(J_err, extent=extent, origin="lower", cmap=plotting.DIVERGING,
                                   vmin=-j_lim, vmax=j_lim)
        axes[0, col].set_title(f"$J - J^*$ at t = {time:g}", loc="left")
        im_u = axes[1, col].imshow(u_err, extent=extent, origin="lower", cmap=plotting.SEQUENTIAL,
                                   vmin=0, vmax=u_lim)
        U, V = u[:, 0].view(grid, grid), u[:, 1].view(grid, grid)
        axes[1, col].quiver(X[skip, skip], Y[skip, skip], U[skip, skip], V[skip, skip],
                            color=plotting.INK, width=0.004, scale=22)
        axes[1, col].set_title(f"$\\|u - u^*\\|$ and learned $u$ at t = {time:g}", loc="left")
        for ax in axes[:, col]:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, ls="--", lw=1.2, color=plotting.INK))
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.grid(False)
    fig.colorbar(im_j, ax=axes[0, :], shrink=0.85, label="cost-to-go error")
    fig.colorbar(im_u, ax=axes[1, :], shrink=0.85, label="control error")
    fig.suptitle("Motion control, n = 2. Dashed circle: ||x|| = T - t; inside it J* = 0 and u* is not unique.",
                 x=0.0, ha="left", fontsize=10, color=plotting.INK_SECONDARY)
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(histories: Dict[int, pd.DataFrame], path: Path, window: int = 50) -> None:
    panels = [("hjb", "HJB residual, mean $r^2$"), ("value_fit", "Cost-to-go fit to simulated cost")]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.6), sharex=True)
    for ax, (column, title) in zip(axes, panels):
        for i, (dim, history) in enumerate(sorted(histories.items())):
            smooth = history[column].rolling(window, min_periods=1).mean()
            ax.plot(history.index, smooth, color=plotting.SERIES[i], label=f"n = {dim}", lw=1.4)
        for boundary in history.index[history.stage.diff().fillna(0) != 0]:
            ax.axvline(boundary, color=plotting.GRID, lw=1.0, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch (stages widen the horizon backward from T)")
        ax.set_title(title, loc="left")
    axes[0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.apply_style()
    runs = collect_runs(MODEL_DIR)
    if not runs:
        raise SystemExit(f"No checkpoints in {MODEL_DIR}; run `python -m motion_control.train` first.")
    rows = []
    for (variant, tag, seed), path in runs.items():
        problem, policy, value, ckpt = load_run(path, MotionControl)
        print(f"evaluating {variant}/{tag} seed {seed}")
        rows.append({"variant": variant, "dim": problem.state_dim, "seed": seed,
                     **evaluate(problem, policy, value), "train_minutes": ckpt["train_seconds"] / 60})
    metrics = pd.DataFrame(rows).sort_values(["variant", "dim", "seed"])
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, float_format="%.6g")
    summary = metrics.groupby(["variant", "dim"])[METRICS + ["train_minutes"]].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(RESULTS_DIR / "summary.csv", index=False, float_format="%.4g")
    print(summary[[f"{m}_mean" for m in METRICS]].to_string(float_format="%.4g"))

    plot_error_vs_dimension(metrics, FIGURE_DIR / "error_vs_dimension.png")
    if ("full", "n2", 0) in runs:
        problem, policy, value, _ = load_run(runs[("full", "n2", 0)], MotionControl)
        plot_slices(problem, policy, value, FIGURE_DIR / "n2_slices.png")
    histories = {int(tag[1:]): pd.read_csv(path.with_suffix(".csv"), index_col="epoch")
                 for (variant, tag, seed), path in runs.items() if variant == "full" and seed == 0}
    if histories:
        plot_training_curves(histories, FIGURE_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
