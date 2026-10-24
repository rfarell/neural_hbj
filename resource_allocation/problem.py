"""Resource allocation problem and its exact solution."""

from typing import Optional

import torch

from neural_hjb import ControlProblem


class ResourceAllocation(ControlProblem):
    """Maximize int_0^T (1 - u) x dt subject to dx/dt = gamma u x, u in [0, 1].

    x is the production rate; a fraction u is reinvested (growing x) and the rest
    is consumed (earning reward). Written as minimization, g = -(1 - u) x and h = 0,
    so J is the negated reward-to-go V.

    Exact solution (bang-bang): reinvest everything until t_s = T - 1/gamma, then
    consume everything. V*(t, x) = x phi(t), with

        phi(t) = T - t                              for t >= t_s,
        phi(t) = exp(gamma (t_s - t)) / gamma       for t <  t_s.

    Args:
        gamma: Growth rate of reinvested production (default: 0.5)
        T: Horizon (default: 4.0)
        x_min, x_max: States are sampled uniformly from [x_min, x_max] (default: 1, 5)
    """

    name = "resource_allocation"
    constraint = "unit_interval"
    state_dim = 1
    control_dim = 1

    def __init__(self, gamma: float = 0.5, T: float = 4.0, x_min: float = 1.0, x_max: float = 5.0) -> None:
        self.gamma = gamma
        self.T = T
        self.x_min = x_min
        self.x_max = x_max

    @property
    def tag(self) -> str:
        return "ra"

    @property
    def switching_time(self) -> float:
        return self.T - 1 / self.gamma

    @property
    def value_scale(self) -> float:
        """Reward of consuming the mean production rate over the whole horizon."""
        return (self.x_min + self.x_max) / 2 * self.T

    def config(self) -> dict:
        return {"gamma": self.gamma, "T": self.T, "x_min": self.x_min, "x_max": self.x_max}

    def dynamics(self, t, x, u):
        return self.gamma * u * x

    def running_cost(self, t, x, u):
        return -(1 - u) * x

    def sample_states(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.x_min + (self.x_max - self.x_min) * torch.rand(n, 1, generator=generator)

    def optimal_control(self, t, x):
        return (t < self.switching_time).to(x.dtype).expand(-1, 1)

    def value_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """phi(t), so that the optimal reward-to-go is V*(t, x) = x phi(t)."""
        reinvest = torch.exp(self.gamma * (self.switching_time - t)) / self.gamma
        return torch.where(t < self.switching_time, reinvest, self.T - t)

    def optimal_cost(self, t, x):
        return -x * self.value_coefficient(t)
