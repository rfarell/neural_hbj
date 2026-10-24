"""Interface for finite-horizon optimal control problems."""

from typing import Optional

import torch


class ControlProblem:
    """Minimize  int_t^T g(s, x, u) ds + h(x(T))  subject to  dx/ds = f(s, x, u).

    The cost-to-go J(t, x) solves the HJB equation

        J_t + min_u [ J_x . f(t, x, u) + g(t, x, u) ] = 0,    J(T, x) = h(x).

    Maximization problems are written as minimization of the negated reward.
    Subclasses define f, g, h, the control set and the training domain, plus the
    exact solution if the learned one is to be scored against it.

    All tensors are batched: t is [B, 1], x is [B, state_dim], u is [B, control_dim].
    """

    name: str
    state_dim: int
    control_dim: int
    constraint: str  # see PolicyNet.CONSTRAINTS
    T: float
    value_scale: float = 1.0  # typical magnitude of J; scales the value network's output

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """f(t, x, u), shape [B, state_dim]."""
        raise NotImplementedError

    def running_cost(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """g(t, x, u), shape [B, 1]."""
        return torch.zeros_like(t)

    def terminal_cost(self, x: torch.Tensor) -> torch.Tensor:
        """h(x), shape [B, 1]."""
        return torch.zeros(x.shape[0], 1, dtype=x.dtype)

    def sample_states(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw n states from the training/test domain, shape [n, state_dim]."""
        raise NotImplementedError

    def optimal_control(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Exact optimal control u*(t, x)."""
        raise NotImplementedError

    def optimal_cost(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Exact cost-to-go J*(t, x)."""
        raise NotImplementedError

    @property
    def tag(self) -> str:
        """Short identifier used in checkpoint file names."""
        return self.name

    def config(self) -> dict:
        """Constructor arguments, stored with checkpoints so the problem can be rebuilt."""
        return {}
