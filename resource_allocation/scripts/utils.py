"""Utility functions and neural network models for resource allocation problem.

This module provides neural network architectures and training utilities
for solving the resource allocation optimal control problem using HJB equations.
"""

import os
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

class UNet(nn.Module):
    """Neural network for approximating the optimal control function u(t,x).
    
    For the resource allocation problem, the control u is constrained to [0,1]
    and represents the fraction of resources allocated to production.
    
    Args:
        state_dim: Dimension of the state space (default: 1)
        hidden_dim: Number of hidden units in each layer (default: 64)
    """
    
    def __init__(self, state_dim: int = 1, hidden_dim: int = 64) -> None:
        super(UNet, self).__init__()
        input_dim = 1 + state_dim  # t and x
        output_dim = 1             # u is scalar in [0,1]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensures output is in (0,1)
        )
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the control u(t,x) constrained to [0,1].
        
        Args:
            t: Time tensor of shape [batch_size, 1] or scalar
            x: State tensor of shape [batch_size, state_dim] or scalar
            
        Returns:
            Control tensor of shape [batch_size, 1] with values in [0,1]
        """
        # Ensure t and x are 2D tensors
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # Shape: [batch_size, 1]
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(1)  # Shape: [1, state_dim]
        elif x.dim() == 1:
            x = x.unsqueeze(1)  # Shape: [batch_size, state_dim]
        tx = torch.cat((t, x), dim=1)
        u = self.net(tx)
        return u

class JNet(nn.Module):
    """Neural network for approximating the cost-to-go function J(t,x).
    
    Args:
        state_dim: Dimension of the state space (default: 1)
        hidden_dim: Number of hidden units in each layer (default: 64)
    """
    
    def __init__(self, state_dim: int = 1, hidden_dim: int = 64) -> None:
        super(JNet, self).__init__()
        input_dim = 1 + state_dim  # t and x
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the cost-to-go J(t,x).
        
        Args:
            t: Time tensor of shape [batch_size, 1] or scalar
            x: State tensor of shape [batch_size, state_dim] or scalar
            
        Returns:
            Cost-to-go tensor of shape [batch_size, 1]
        """
        # Ensure t and x are 2D tensors
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # Shape: [batch_size, 1]
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(1)  # Shape: [1, state_dim]
        elif x.dim() == 1:
            x = x.unsqueeze(1)  # Shape: [batch_size, state_dim]
        tx = torch.cat((t, x), dim=1)
        return self.net(tx)

def augmented_dynamics(
    t: torch.Tensor,
    y: torch.Tensor,
    u_net: UNet,
    gamma: float
) -> torch.Tensor:
    """Augmented system dynamics for simultaneous state and cost integration.
    
    The augmented state y = [x, c] evolves according to:
    - dx/dt = gamma * u * x (production dynamics)
    - dc/dt = (1 - u) * x (cumulative cost)
    
    Args:
        t: Current time (scalar tensor)
        y: Augmented state tensor of shape [batch_size, 2] where
           y[:, 0] = x (production level)
           y[:, 1] = c (cumulative cost)
        u_net: Neural network model for control
        gamma: Production growth parameter
        
    Returns:
        Augmented state derivative tensor of shape [batch_size, 2]
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)  # Shape: [1, 2]
    x = y[:, 0:1]  # Shape: [batch_size, 1]
    c = y[:, 1:2]  # Shape: [batch_size, 1] (not used in dynamics)

    # Prepare t_tensor to match the batch size of x
    t_tensor = t * torch.ones_like(x)  # Shape: [batch_size, 1]

    # Compute control u(t, x)
    u = u_net(t_tensor, x)  # Shape: [batch_size, 1]

    # Compute dynamics
    dxdt = gamma * u * x  # Shape: [batch_size, 1]
    dcdt = (1 - u) * x    # Shape: [batch_size, 1]

    dy_dt = torch.cat([dxdt, dcdt], dim=1)  # Shape: [batch_size, 2]

    return dy_dt.squeeze(0) if dy_dt.shape[0] == 1 else dy_dt

def TrainStage(
    T_start: float,
    T_end: float,
    u_net: UNet,
    j_net: JNet,
    optimizer: optim.Optimizer,
    num_epochs: int = 1000,
    state_dim: int = 1,
    gamma: float = 0.5
) -> List[float]:
    """Train neural networks for a single time stage [T_start, T_end].
    
    Args:
        T_start: Start time of the stage
        T_end: End time of the stage (terminal time)
        u_net: Control network model
        j_net: Cost-to-go network model (lowercase for consistency)
        optimizer: PyTorch optimizer for both networks
        num_epochs: Number of training epochs (default: 1000)
        state_dim: Dimension of the state space (default: 1)
        gamma: Production growth parameter (default: 0.5)
        
    Returns:
        List of training losses for each epoch
    """
    training_losses = []
    for epoch in range(num_epochs):
        # Sample minibatch of initial points at time T_start
        M = 128
        t_j = torch.ones(M, 1) * T_start
        x_j = torch.rand(M, state_dim) * 4 + 1  # x_j in [1,5]

        # Sample minibatch of terminal points at time T_end
        M_T = 128
        t_T = torch.ones(M_T, 1) * T_end
        x_T = torch.rand(M_T, state_dim) * 4 + 1  # x_k in [1,5]

        # Ensure t_j and x_j require gradients
        t_j = t_j.clone().detach().requires_grad_(True)
        x_j = x_j.clone().detach().requires_grad_(True)

        # Compute residuals
        l1, l2, l3 = compute_residuals(t_j, x_j, u_net, j_net, gamma)

        # Terminal condition (no terminal cost)
        J_T = j_net(t_T, x_T)
        h_T = torch.zeros_like(J_T)
        l4 = ((J_T - h_T) ** 2).mean()

        # Simulate augmented trajectory from t_j to T_end
        t_span = torch.tensor([T_start, T_end]).to(t_j.device)
        y0 = torch.cat([x_j, torch.zeros_like(x_j)], dim=1)  # Initial augmented state [x_j, c=0]
        y_t = odeint(lambda t, y: augmented_dynamics(t, y, u_net, gamma), y0, t_span)
        y_Tj = y_t[-1]  # Shape: [batch_size, 2]
        x_Tj = y_Tj[:, 0:1]  # Final state x(T)
        C_j = y_Tj[:, 1:2]   # Cumulative cost from t_j to T_end

        # Compute l5 and l6
        l5 = (C_j ** 2).mean()
        J = j_net(t_j, x_j)
        l6 = ((J - C_j) ** 2).mean()

        # Total loss
        loss = l1 + l2 + l3 + l4 - l5 + l6

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        training_losses.append(loss.item())

        # Print losses every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    return training_losses

def compute_residuals(
    t: torch.Tensor,
    x: torch.Tensor,
    u_net: UNet,
    j_net: JNet,
    gamma: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute HJB equation residuals for the resource allocation problem.
    
    Computes three residual terms:
    - l1: HJB equation residual
    - l2: Gradient consistency residual
    - l3: Second-order time derivative residual
    
    Args:
        t: Time tensor of shape [batch_size, 1]
        x: State tensor of shape [batch_size, 1]
        u_net: Control network model
        j_net: Cost-to-go network model
        gamma: Production growth parameter
        
    Returns:
        Tuple of (l1, l2, l3) loss terms
    """
    # Ensure t and x require gradients
    t = t.clone().detach().requires_grad_(True)  # [batch_size, 1]
    x = x.clone().detach().requires_grad_(True)  # [batch_size, 1]

    # Forward pass
    u = u_net(t, x)          # [batch_size, 1]
    J = j_net(t, x)          # [batch_size, 1]

    # Compute dynamics
    f = gamma * u * x        # [batch_size, 1]

    # Compute running cost (Note: we maximize (1 - u) x, so g = - (1 - u) x)
    g = - (1 - u) * x        # [batch_size, 1]

    # Compute gradients
    grad_J = torch.autograd.grad(J.sum(), x, create_graph=True)[0]     # [batch_size, 1]
    grad_J_t = torch.autograd.grad(J.sum(), t, create_graph=True)[0]   # [batch_size, 1]

    # Compute l1 residual: HJB equation residual
    HJB_residual = g + grad_J_t + grad_J * f
    l1 = (HJB_residual ** 2).mean()

    # Compute l2 residual: Gradient w.r.t x
    # Since the state is 1D, grad_J_xx is scalar
    grad_J_xx = torch.autograd.grad(grad_J.sum(), x, create_graph=True)[0]  # [batch_size, 1]
    l2_residual = grad_J_xx + grad_J * gamma * u
    l2 = (l2_residual ** 2).mean()

    # Compute l3 residual: Second derivative w.r.t t
    grad_J_tt = torch.autograd.grad(grad_J_t.sum(), t, create_graph=True)[0]  # [batch_size, 1]
    l3_residual = grad_J_tt + grad_J_t * f
    l3 = (l3_residual ** 2).mean()

    return l1, l2, l3

def save_models(u_net: UNet, j_net: JNet, output_dir: str, stage: int) -> None:
    """Save model checkpoints for a specific training stage.
    
    Args:
        u_net: Control network model to save
        j_net: Cost-to-go network model to save
        output_dir: Directory to save the models
        stage: Training stage number
    """
    u_net_path = os.path.join(output_dir, f'u_net_stage_{stage}.pth')
    j_net_path = os.path.join(output_dir, f'j_net_stage_{stage}.pth')
    torch.save(u_net.state_dict(), u_net_path)
    torch.save(j_net.state_dict(), j_net_path)

def compute_ground_truth(
    t: torch.Tensor,
    T: float,
    gamma: float
) -> torch.Tensor:
    """Compute the analytical optimal control for the resource allocation problem.
    
    The optimal control has a bang-bang structure:
    - u*(t) = 1 for t < T - 1/gamma (maximum production)
    - u*(t) = 0 for t >= T - 1/gamma (maximum consumption)
    
    Args:
        t: Time tensor or scalar
        T: Terminal time
        gamma: Production growth parameter
        
    Returns:
        Optimal control value(s) at time t
    """
    # Compute the switching time
    switching_time = T - 1 / gamma
    
    # Ensure t is a torch tensor
    t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
    
    # Compute u_star using PyTorch tensor operations
    u_star = torch.where(t < switching_time, torch.tensor(1.0), torch.tensor(0.0))
    
    return u_star

def load_models(model_dir: str, state_dim: int) -> Tuple[UNet, JNet]:
    """Load trained models from checkpoint files.
    
    Loads the models from stage 0 (final stage) which represents
    the complete time horizon [0, T].
    
    Args:
        model_dir: Directory containing the model checkpoints
        state_dim: Dimension of the state space
        
    Returns:
        Tuple of (u_net, j_net) loaded models in evaluation mode
    """
    u_net = UNet(state_dim=state_dim)
    j_net = JNet(state_dim=state_dim)
    # Load the models from the last stage (stage 0)
    u_net.load_state_dict(torch.load(os.path.join(model_dir, 'u_net_stage_0.pth')))
    j_net.load_state_dict(torch.load(os.path.join(model_dir, 'j_net_stage_0.pth')))
    u_net.eval()
    j_net.eval()
    return u_net, j_net
