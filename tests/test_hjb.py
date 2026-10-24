"""The exact solutions must satisfy the HJB losses and the closed-loop simulation."""

import pytest
import torch

from motion_control.problem import MotionControl
from neural_hjb import PolicyNet, collocation_losses, hamiltonian, rollout
from resource_allocation.problem import ResourceAllocation


class Exact(torch.nn.Module):
    """Wrap a problem's exact u* or J* as a network."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t, x):
        return self.fn(t, x)


def motion_control_points(problem, n=512):
    """Samples away from the kink ||x|| = T - t of J* and the singularity of u* at x = 0."""
    g = torch.Generator().manual_seed(0)
    t = problem.T * torch.rand(n, 1, generator=g)
    x = problem.sample_states(n, generator=g)
    keep = ((x.norm(dim=1, keepdim=True) - (problem.T - t)).abs() > 0.05).squeeze(1)
    return t[keep], x[keep]


def resource_allocation_points(problem, n=512):
    """Samples away from the switching time, where u* jumps."""
    g = torch.Generator().manual_seed(0)
    t = problem.T * torch.rand(n, 1, generator=g)
    x = problem.sample_states(n, generator=g)
    keep = ((t - problem.switching_time).abs() > 0.05).squeeze(1)
    return t[keep], x[keep]


@pytest.mark.parametrize("state_dim", [1, 2, 8])
def test_motion_control_exact_solution_has_zero_hjb_loss(state_dim):
    problem = MotionControl(state_dim=state_dim)
    t, x = motion_control_points(problem)
    losses = collocation_losses(problem, Exact(problem.optimal_control), Exact(problem.optimal_cost), t, x)
    for name, loss in losses.items():
        assert loss.item() < 1e-8, name


def test_resource_allocation_exact_solution_has_zero_hjb_loss():
    problem = ResourceAllocation()
    t, x = resource_allocation_points(problem)
    losses = collocation_losses(problem, Exact(problem.optimal_control), Exact(problem.optimal_cost), t, x)
    for name, loss in losses.items():
        assert loss.item() < 1e-8, name


def test_resource_allocation_wrong_sign_reward_is_detected():
    """Regression: the original code used the running reward with the wrong sign."""

    class WrongSign(ResourceAllocation):
        def running_cost(self, t, x, u):
            return -super().running_cost(t, x, u)

    problem = WrongSign()
    t, x = resource_allocation_points(problem)
    losses = collocation_losses(problem, Exact(problem.optimal_control), Exact(problem.optimal_cost), t, x)
    assert losses["hjb"].item() > 1.0


@pytest.mark.parametrize(
    "problem, points",
    [
        (MotionControl(state_dim=3), motion_control_points),
        (ResourceAllocation(), resource_allocation_points),
    ],
)
def test_exact_policy_rollout_achieves_exact_cost(problem, points):
    t, x = points(problem)
    if problem.name == "motion_control":
        # u* = -x/||x|| chatters once it reaches the origin, so start where the origin is out of reach.
        keep = (x.norm(dim=1, keepdim=True) > problem.T - t).squeeze(1)
        t, x = t[keep], x[keep]
    _, _, cost = rollout(problem, Exact(problem.optimal_control), t, x, steps=400)
    torch.testing.assert_close(cost, problem.optimal_cost(t, x), rtol=2e-3, atol=2e-3)


def test_policy_respects_control_constraints():
    torch.manual_seed(0)
    t, x = torch.rand(256, 1), 10 * torch.randn(256, 3)
    ball = PolicyNet(3, 3, "unit_ball")
    assert (ball(t, x).norm(dim=1) <= 1 + 1e-6).all()
    interval = PolicyNet(3, 1, "unit_interval")
    u = interval(t, x)
    assert ((u >= 0) & (u <= 1)).all()


@pytest.mark.parametrize(
    "problem, points",
    [
        (MotionControl(state_dim=4), motion_control_points),
        (ResourceAllocation(), resource_allocation_points),
    ],
)
def test_exact_control_minimizes_hamiltonian(problem, points):
    """u* = argmin_u H(t, x, u, J*_x), the minimization in the HJB equation."""
    t, x = points(problem)
    x = x.requires_grad_(True)
    J_x, = torch.autograd.grad(problem.optimal_cost(t, x).sum(), x)
    x = x.detach()
    H_star = hamiltonian(problem, t, x, problem.optimal_control(t, x), J_x)
    torch.manual_seed(0)
    for _ in range(20):
        u = PolicyNet(problem.state_dim, problem.control_dim, problem.constraint)(t, 3 * torch.randn_like(x))
        assert (H_star <= hamiltonian(problem, t, x, u, J_x) + 1e-5).all()
