"""Policy and value networks u(t, x) and J(t, x)."""

import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Sequential:
    """Two tanh hidden layers. tanh keeps the higher derivatives used by the HJB losses smooth."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class PolicyNet(nn.Module):
    """Feedback control u(t, x), mapped into the problem's control set.

    Args:
        state_dim: Dimension of the state x
        control_dim: Dimension of the control u
        constraint: 'unit_ball' (||u|| <= 1, by radial projection) or
            'unit_interval' (u in (0, 1), by a sigmoid)
        hidden_dim: Width of the hidden layers (default: 64)
    """

    CONSTRAINTS = ("unit_ball", "unit_interval")

    def __init__(self, state_dim: int, control_dim: int, constraint: str, hidden_dim: int = 64) -> None:
        super().__init__()
        if constraint not in self.CONSTRAINTS:
            raise ValueError(f"constraint must be one of {self.CONSTRAINTS}, got {constraint!r}")
        self.constraint = constraint
        self.net = mlp(1 + state_dim, control_dim, hidden_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map t [B, 1] and x [B, state_dim] to an admissible control [B, control_dim]."""
        u = self.net(torch.cat((t, x), dim=1))
        if self.constraint == "unit_ball":
            return u / torch.clamp(u.norm(dim=1, keepdim=True), min=1.0)
        return torch.sigmoid(u)


class ValueNet(nn.Module):
    """Cost-to-go J(t, x) = output_scale * MLP(t, x).

    With output_scale set to the typical size of the cost-to-go, the MLP works at
    O(1) regardless of the problem's units. Without it, Adam needs thousands of
    steps just to grow the output weights when J is large (e.g. J up to ~32 for
    8-D motion control), and the value stalls near a constant.

    Args:
        state_dim: Dimension of the state x
        hidden_dim: Width of the hidden layers (default: 64)
        output_scale: Fixed multiplier on the MLP output (default: 1.0)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, output_scale: float = 1.0) -> None:
        super().__init__()
        self.net = mlp(1 + state_dim, 1, hidden_dim)
        self.output_scale = output_scale

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Map t [B, 1] and x [B, state_dim] to J [B, 1]."""
        return self.output_scale * self.net(torch.cat((t, x), dim=1))
