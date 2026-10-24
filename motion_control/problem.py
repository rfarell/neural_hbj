"""Motion control problem and its exact solution."""

from typing import Optional

import torch

from neural_hjb import ControlProblem


class MotionControl(ControlProblem):
    """Minimize ||x(T)||^2 subject to dx/dt = u, ||u|| <= 1, x in R^n.

    Exact solution: move straight at the origin at full speed, u* = -x/||x||, so

        J*(t, x) = max(0, ||x|| - (T - t))^2.

    Inside the ball ||x|| <= T - t the origin is reachable and J* = 0; there any
    control that reaches the origin by T is optimal, so u* is not unique.

    Args:
        state_dim: Dimension n of the state (default: 2)
        T: Horizon (default: 1.0)
        half_width: States are sampled uniformly from [-half_width, half_width]^n (default: 2.0)
    """

    name = "motion_control"
    constraint = "unit_ball"

    def __init__(self, state_dim: int = 2, T: float = 1.0, half_width: float = 2.0) -> None:
        self.state_dim = state_dim
        self.control_dim = state_dim
        self.T = T
        self.half_width = half_width

    @property
    def tag(self) -> str:
        return f"n{self.state_dim}"

    @property
    def value_scale(self) -> float:
        """Mean terminal cost E||x||^2 over the sampling box."""
        return self.state_dim * self.half_width ** 2 / 3

    def config(self) -> dict:
        return {"state_dim": self.state_dim, "T": self.T, "half_width": self.half_width}

    def dynamics(self, t, x, u):
        return u

    def terminal_cost(self, x):
        return x.pow(2).sum(dim=1, keepdim=True)

    def sample_states(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return (2 * torch.rand(n, self.state_dim, generator=generator) - 1) * self.half_width

    def optimal_control(self, t, x):
        return -x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def optimal_cost(self, t, x):
        return torch.clamp(x.norm(dim=1, keepdim=True) - (self.T - t), min=0).pow(2)
