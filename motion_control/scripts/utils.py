"""Utility functions and neural network models for solving Hamilton-Jacobi-Bellman equations.

This module provides neural network architectures (UNet and JNet) and training utilities
for solving optimal control problems using the HJB equation framework.
"""

import os
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

class UNet(nn.Module):
    """Neural network for approximating the optimal control function u(t,x).
    
    The network enforces the control constraint ||u|| <= 1 through normalization.
    
    Args:
        state_dim: Dimension of the state space (default: 2)
        hidden_dim: Number of hidden units in each layer (default: 64)
    """
    
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64) -> None:
        super(UNet, self).__init__()
        input_dim = 1 + state_dim  # t and x
        output_dim = state_dim     # u has same dimension as x
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute the control u(t,x) with constraint ||u|| <= 1.
        
        Args:
            t: Time tensor of shape [batch_size, 1]
            x: State tensor of shape [batch_size, state_dim]
            
        Returns:
            Control tensor of shape [batch_size, state_dim] with ||u|| <= 1
        """
        tx = torch.cat((t, x), dim=1)
        u = self.net(tx)
        # Enforce control constraint ||u|| <= 1
        u_norm = torch.norm(u, dim=1, keepdim=True)
        u_clipped = u / torch.clamp(u_norm, min=1.0)
        return u_clipped

class JNet(nn.Module):
    """Neural network for approximating the cost-to-go function J(t,x).
    
    Args:
        state_dim: Dimension of the state space (default: 2)
        hidden_dim: Number of hidden units in each layer (default: 64)
    """
    
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64) -> None:
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
            t: Time tensor of shape [batch_size, 1]
            x: State tensor of shape [batch_size, state_dim]
            
        Returns:
            Cost-to-go tensor of shape [batch_size, 1]
        """
        tx = torch.cat((t, x), dim=1)
        return self.net(tx)

def dynamics(t: torch.Tensor, x: torch.Tensor, u_net: UNet) -> torch.Tensor:
    """System dynamics dx/dt = u(t,x) for ODE integration.
    
    Args:
        t: Current time (scalar tensor)
        x: State tensor of shape [batch_size, state_dim]
        u_net: Neural network model for control
        
    Returns:
        State derivative tensor of shape [batch_size, state_dim]
    """
    t_tensor = t * torch.ones(x.shape[0], 1, device=x.device)
    u = u_net(t_tensor, x)
    return u

def TrainStage(
    T_start: float,
    T_end: float,
    u_net: UNet,
    J_net: JNet,
    optimizer: optim.Optimizer,
    num_epochs: int = 1000,
    state_dim: int = 2
) -> List[float]:
    """Train neural networks for a single time stage [T_start, T_end].
    
    Args:
        T_start: Start time of the stage
        T_end: End time of the stage (terminal time)
        u_net: Control network model
        J_net: Cost-to-go network model
        optimizer: PyTorch optimizer for both networks
        num_epochs: Number of training epochs (default: 1000)
        state_dim: Dimension of the state space (default: 2)
        
    Returns:
        List of training losses for each epoch
    """
    training_losses = []
    for epoch in range(num_epochs):
        # Sample minibatch of initial points at time T_start
        M = 128
        t_j = torch.ones(M, 1) * T_start
        x_j = torch.rand(M, state_dim) * 4 - 2  # x_j in [-2,2]^state_dim
        
        # Sample minibatch of terminal points at time T_end
        M_T = 128
        t_T = torch.ones(M_T, 1) * T_end
        x_T = torch.rand(M_T, state_dim) * 4 - 2  # x_k in [-2,2]^state_dim
        
        # Ensure t_j and x_j require gradients
        t_j = t_j.clone().detach().requires_grad_(True)
        x_j = x_j.clone().detach().requires_grad_(True)
        
        # Compute residuals
        l1, l2, l3 = compute_residuals(t_j, x_j, u_net, J_net)
        
        # Compute terminal loss l4
        J_T = J_net(t_T, x_T)
        h_T = (x_T ** 2).sum(dim=1, keepdim=True)
        l4 = ((J_T - h_T) ** 2).mean()
        
        # Compute cost-to-go losses l5 and l6
        t_span = torch.tensor([T_start, T_end]).to(t_j.device)
        x_t = odeint(lambda t, x: dynamics(t, x, u_net), x_j, t_span)
        x_Tj = x_t[-1]  # x_Tj shape: [M, state_dim]
        h_Tj = (x_Tj ** 2).sum(dim=1, keepdim=True)
        J = J_net(t_j, x_j)
        l5 = (h_Tj ** 2).mean()
        l6 = ((J - h_Tj) ** 2).mean()
        
        # Total loss
        loss = l1 + l2 + l3 + l4 + l5 + l6
        
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
    J_net: JNet
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute HJB equation residuals for the loss function.
    
    Computes three residual terms:
    - l1: First-order optimality condition residual
    - l2: Second-order time derivative residual
    - l3: HJB equation residual
    
    Args:
        t: Time tensor of shape [batch_size, 1]
        x: State tensor of shape [batch_size, state_dim]
        u_net: Control network model
        J_net: Cost-to-go network model
        
    Returns:
        Tuple of (l1, l2, l3) loss terms
    """
    # Ensure t and x require gradients
    t = t.clone().detach().requires_grad_(True)  # [batch_size, 1]
    x = x.clone().detach().requires_grad_(True)  # [batch_size, state_dim]

    # Forward pass
    u = u_net(t, x)  # [batch_size, state_dim]
    J = J_net(t, x)  # [batch_size, 1]

    # Compute gradients
    grad_J = torch.autograd.grad(J.sum(), x, create_graph=True)[0]     # [batch_size, state_dim]
    grad_J_t = torch.autograd.grad(J.sum(), t, create_graph=True)[0]  # [batch_size, 1]

    # Compute grad_J_xt: gradient of grad_J_t w.r.t x
    grad_J_xt = torch.autograd.grad(grad_J_t.sum(), x, create_graph=True)[0]  # [batch_size, state_dim]

    # Compute Hessian-vector product grad_J_xx_u = ∇^2_x J * u
    grad_J_xx_u = torch.autograd.grad(grad_J, x, grad_outputs=u, create_graph=True)[0]  # [batch_size, state_dim]

    # Compute Jacobian-vector product grad_u_T_grad_J = (∇_x u)^T * grad_J
    grad_u_T_grad_J = torch.autograd.grad(u, x, grad_outputs=grad_J, create_graph=True)[0]  # [batch_size, state_dim]

    # Compute l1 residual
    l1_residual = grad_J_xt + grad_J_xx_u + grad_u_T_grad_J  # [batch_size, state_dim]
    l1 = (l1_residual.norm(2, dim=1) ** 2).mean()

    # Compute second derivative of J w.r.t t
    grad_J_tt = torch.autograd.grad(grad_J_t.sum(), t, create_graph=True)[0]  # [batch_size, 1]

    # Compute (∇xt J) ⋅ u
    grad_J_xt_u = (grad_J_xt * u).sum(dim=1, keepdim=True)  # [batch_size, 1]

    # Compute l2 residual
    l2_residual = grad_J_tt + grad_J_xt_u  # [batch_size, 1]
    l2 = (l2_residual ** 2).mean()

    # Compute grad_J_t + grad_J_x^T u (HJB equation)
    grad_J_t_u = grad_J_t + (grad_J * u).sum(dim=1, keepdim=True)  # [batch_size, 1]

    # Compute l3 residual
    l3_residual = grad_J_t_u  # [batch_size, 1]
    l3 = (l3_residual ** 2).mean()

    return l1, l2, l3

def save_models(u_net: UNet, J_net: JNet, output_dir: str, stage: int) -> None:
    """Save model checkpoints for a specific training stage.
    
    Args:
        u_net: Control network model to save
        J_net: Cost-to-go network model to save
        output_dir: Directory to save the models
        stage: Training stage number
    """
    u_net_path = os.path.join(output_dir, f'u_net_stage_{stage}.pth')
    J_net_path = os.path.join(output_dir, f'J_net_stage_{stage}.pth')
    torch.save(u_net.state_dict(), u_net_path)
    torch.save(J_net.state_dict(), J_net_path)

def compute_ground_truth(
    t: torch.Tensor,
    x: torch.Tensor,
    T: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical solution for the motion control problem.
    
    For the quadratic cost h(x) = ||x||^2, the optimal control is:
    - u*(t,x) = -x/||x|| when ||x|| > 0
    - u*(t,x) = 0 when ||x|| = 0
    
    The optimal cost-to-go is:
    - J*(t,x) = (||x|| - (T-t))^2 when ||x|| > T-t
    - J*(t,x) = 0 otherwise
    
    Args:
        t: Time tensor of shape [batch_size, 1]
        x: State tensor of shape [batch_size, state_dim]
        T: Terminal time
        
    Returns:
        Tuple of (u_star, J_star):
        - u_star: Optimal control of shape [batch_size, state_dim]
        - J_star: Optimal cost-to-go of shape [batch_size, 1]
    """
    x_norm = torch.norm(x, dim=1, keepdim=True)  # L2 norm of x(t)
    u_star = torch.zeros_like(x)                 # Initialize u^*(t)
    non_zero_norm_indices = (x_norm > 0).squeeze()
    if non_zero_norm_indices.sum() > 0:
        x_non_zero = x[non_zero_norm_indices]
        x_norm_non_zero = x_norm[non_zero_norm_indices]
        u_star[non_zero_norm_indices, :] = -x_non_zero / x_norm_non_zero

    # Compute the ground truth J^*(t,x)
    J_star = torch.zeros_like(x_norm)
    condition_indices = (x_norm > (T - t))
    J_star[condition_indices] = (x_norm[condition_indices] - (T - t[condition_indices])) ** 2

    return u_star, J_star

def load_models(model_dir: str, state_dim: int) -> Tuple[UNet, JNet]:
    """Load trained models from checkpoint files.
    
    Loads the models from stage 0 (final stage) which represents
    the complete time horizon [0, T].
    
    Args:
        model_dir: Directory containing the model checkpoints
        state_dim: Dimension of the state space
        
    Returns:
        Tuple of (u_net, J_net) loaded models in evaluation mode
    """
    u_net = UNet(state_dim=state_dim)
    J_net = JNet(state_dim=state_dim)
    # Load the models from the last stage (stage 0)
    u_net.load_state_dict(torch.load(os.path.join(model_dir, 'u_net_stage_0.pth')))
    J_net.load_state_dict(torch.load(os.path.join(model_dir, 'J_net_stage_0.pth')))
    u_net.eval()
    J_net.eval()
    return u_net, J_net
