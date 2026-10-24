"""HJB losses, closed-loop rollouts and the staged training loop.

The losses are built generically from a ControlProblem's f, g and h, so no
problem needs hand-derived residuals. With u = policy(t, x):

    hjb          r(t, x)^2, where r = J_t + J_x . f(t, x, u) + g(t, x, u)
    hjb_dt       (dr/dt)^2      derivatives of the residual with u held fixed (envelope
    hjb_dx       ||dr/dx||^2    theorem); like r they vanish at the exact solution
    hamiltonian  H = J_x . f(t, x, u) + g(t, x, u) with J held fixed: pushes u toward
                 argmin_u H, the minimization in the HJB equation
    terminal     (J(T, x) - h(x))^2
    value_fit    (J(t, x) - C(t, x))^2, C the simulated cost of the policy (held fixed)
    policy       C(t, x), the cost the policy actually achieves, differentiated through the ODE

The residual terms train only the value network. Letting them train the policy too
(as the first version of this repo did) is unsound: for the exact J, r(u) =
H(u) - min H >= 0, but while J is inaccurate r can be negative for every u, and
minimizing r^2 then pushes the policy toward controls with a larger Hamiltonian.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torchdiffeq import odeint

from .networks import PolicyNet, ValueNet
from .problem import ControlProblem

HJB_TERMS = ("hjb", "hjb_dt", "hjb_dx", "hamiltonian")

# Loss terms used by each training variant. 'full' is the method; the others are ablations.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "full": ("hjb", "hjb_dt", "hjb_dx", "hamiltonian", "terminal", "value_fit", "policy"),
    "no_hamiltonian": ("hjb", "hjb_dt", "hjb_dx", "terminal", "value_fit", "policy"),
    "no_hjb_grad": ("hjb", "hamiltonian", "terminal", "value_fit", "policy"),
    "no_hjb": ("terminal", "value_fit", "policy"),
}


def _grad(output: torch.Tensor, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Gradients of output.sum() that stay differentiable; zeros for unused inputs."""
    return torch.autograd.grad(
        output.sum(), inputs, create_graph=True, allow_unused=True, materialize_grads=True
    )


def hamiltonian(problem: ControlProblem, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor,
                J_x: torch.Tensor) -> torch.Tensor:
    """H(t, x, u, J_x) = J_x . f(t, x, u) + g(t, x, u), shape [B, 1]."""
    return (J_x * problem.dynamics(t, x, u)).sum(dim=1, keepdim=True) + problem.running_cost(t, x, u)


def hjb_residual(
    problem: ControlProblem,
    policy: torch.nn.Module,
    value: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """HJB residual r = J_t + H(t, x, u, J_x) at u = policy(t, x), with the pieces it is built from.

    Returns:
        Dict with r [B, 1], the control u [B, control_dim] (still attached to the policy),
        J_x [B, state_dim], and the leaf tensors t and x that r was built from
    """
    t = t.detach().requires_grad_(True)
    x = x.detach().requires_grad_(True)
    u = policy(t, x)
    J = value(t, x)
    J_t, J_x = _grad(J, (t, x))
    r = J_t + hamiltonian(problem, t, x, u.detach(), J_x)
    return {"r": r, "u": u, "J_x": J_x, "t": t, "x": x}


def collocation_losses(
    problem: ControlProblem,
    policy: torch.nn.Module,
    value: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """HJB losses at collocation points (see the module docstring)."""
    res = hjb_residual(problem, policy, value, t, x)
    r_t, r_x = _grad(res["r"], (res["t"], res["x"]))
    H = hamiltonian(problem, res["t"], res["x"], res["u"], res["J_x"].detach())
    return {
        "hjb": res["r"].pow(2).mean(),
        "hjb_dt": r_t.pow(2).mean(),
        "hjb_dx": r_x.pow(2).sum(dim=1).mean(),
        "hamiltonian": H.mean(),
    }


def rollout(
    problem: ControlProblem,
    policy: torch.nn.Module,
    t0: torch.Tensor,
    x0: torch.Tensor,
    steps: int = 20,
    num_points: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate the closed loop dx/dt = f(t, x, policy(t, x)) from (t0, x0) to T.

    Each sample starts at its own time, so time is rescaled to s in [0, 1] with
    t = t0 + s (T - t0), and the running cost is integrated alongside the state.
    Fixed-step RK4 with `steps` steps per trajectory.

    Args:
        t0: Start times, shape [B, 1]
        x0: Start states, shape [B, state_dim]
        steps: RK4 steps per trajectory
        num_points: Number of evenly spaced points of s in [0, 1] to return

    Returns:
        Tuple of (t [num_points, B, 1], x [num_points, B, state_dim], cost [B, 1]),
        where cost is the running cost plus h(x(T)).
    """
    horizon = problem.T - t0

    def augmented(s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = y[:, :-1]
        t = t0 + s * horizon
        u = policy(t, x)
        return torch.cat(
            (problem.dynamics(t, x, u) * horizon, problem.running_cost(t, x, u) * horizon), dim=1
        )

    y0 = torch.cat((x0, torch.zeros_like(t0)), dim=1)
    s = torch.linspace(0.0, 1.0, num_points)
    y = odeint(augmented, y0, s, method="rk4", options={"step_size": 1.0 / steps})
    x = y[..., :-1]
    t = t0 + s.view(-1, 1, 1) * horizon
    cost = y[-1, :, -1:] + problem.terminal_cost(x[-1])
    return t, x, cost


def train_staged(
    problem: ControlProblem,
    policy: PolicyNet,
    value: ValueNet,
    terms: Sequence[str],
    stages: int = 5,
    epochs_per_stage: int = 1000,
    batch_size: int = 128,
    lr: float = 1e-3,
    rollout_steps: int = 20,
    log_every: Optional[int] = 250,
) -> List[Dict[str, float]]:
    """Train backward in time over horizons [t_k, T], t_k = T k / stages, k = stages-1, ..., 0.

    Each epoch samples t uniformly in the current horizon [t_k, T] and x from the
    problem's domain, and minimizes the unweighted sum of the selected loss terms.

    Returns:
        One dict of loss values per epoch
    """
    unknown = set(terms) - set(VARIANTS["full"])
    if unknown:
        raise ValueError(f"unknown loss terms: {sorted(unknown)}")
    optimizer = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=lr)
    history = []
    start = time.time()
    for k in reversed(range(stages)):
        t_start = problem.T * k / stages
        for epoch in range(epochs_per_stage):
            t = t_start + (problem.T - t_start) * torch.rand(batch_size, 1)
            x = problem.sample_states(batch_size)
            losses = {}
            if any(term in HJB_TERMS for term in terms):
                losses.update(collocation_losses(problem, policy, value, t, x))

            x_T = problem.sample_states(batch_size)
            t_T = torch.full((batch_size, 1), problem.T)
            losses["terminal"] = (value(t_T, x_T) - problem.terminal_cost(x_T)).pow(2).mean()

            _, _, cost = rollout(problem, policy, t, x, steps=rollout_steps)
            losses["policy"] = cost.mean()
            # The value regresses onto the simulated cost; detaching keeps this term
            # from pushing the policy toward whatever cost the value currently predicts.
            losses["value_fit"] = (value(t, x) - cost.detach()).pow(2).mean()

            loss = sum(losses[term] for term in terms)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history.append({"stage": k, "loss": loss.item(), **{name: v.item() for name, v in losses.items()}})
            if log_every and epoch % log_every == 0:
                detail = "  ".join(f"{term}={losses[term].item():.3g}" for term in terms)
                print(f"[{problem.tag}] stage {k} [{t_start:.2f}, {problem.T:.2f}] epoch {epoch}: "
                      f"{detail}  ({time.time() - start:.0f}s)", flush=True)
    return history
