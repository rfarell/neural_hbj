# show_various_paths.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import UNet, JNet, load_models
from torchdiffeq import odeint
import os

def main():
    # Configuration Parameters
    gamma = 0.5
    T = 4.0
    state_dim = 1
    model_dir = '../models/resource_allocation'
    plot_dir = '../results/resource_allocation/plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Load the trained models
    u_net, j_net = load_models(model_dir, state_dim)
    u_net.eval()
    j_net.eval()

    # Define a set of initial conditions
    initial_conditions = [1.0, 2.0, 3.0, 4.0]  # Example initial x0 values

    # Define time points for simulation
    num_time_steps = 100
    t = torch.linspace(0, T, steps=num_time_steps)

    # Prepare figure
    plt.figure(figsize=(10, 6))

    # Plot analytical optimal control u*(t)
    u_star = compute_optimal_control(t, gamma, T)
    plt.plot(t.numpy(), u_star.numpy(), label=r'Optimal Control $u^*(t)$', color='black', linestyle='--', linewidth=2)

    # Colors for different initial conditions
    colors = ['blue', 'green', 'red', 'purple']

    for idx, x0 in enumerate(initial_conditions):
        # Initial augmented state [x0, c=0]
        y0 = torch.tensor([x0, 0.0], dtype=torch.float32)

        # Define augmented dynamics for integration
        augmented_dynamics_func = lambda t, y: augmented_dynamics(t, y, u_net, gamma)

        # Integrate the augmented dynamics
        y_t = odeint(augmented_dynamics_func, y0, t)
        # y_t shape: [num_time_steps, 2]

        # Extract control u(t) from the augmented state
        # To extract u(t), we need to compute it at each time step using the neural network
        # Alternatively, since u(t) is part of the dynamics, but in our implementation, it's not stored
        # Therefore, we'll recompute u(t) from x(t)
        x_t = y_t[:, 0]
        u_t = []
        for ti, xi in zip(t, x_t):
            ti_tensor = ti * torch.ones(1, 1)  # Shape: [1, 1]
            xi_tensor = xi.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]
            with torch.no_grad():
                u = u_net(ti_tensor, xi_tensor)
            u_t.append(u.item())
        u_t = np.array(u_t)

        # Plot the neural network's control u(t)
        plt.plot(t.numpy(), u_t, label=f'NN Control $u(t)$, $x_0$={x0}', color=colors[idx], linewidth=2)

    # Plot styling
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel('Control $u(t)$', fontsize=14)
    plt.title('Control Paths for Resource Allocation Problem', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(-0.1, 1.1)  # Since u(t) is bounded between 0 and 1
    plt.xlim(0, T)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'control_paths_resource_allocation.png'), dpi=300)
    plt.show()

def augmented_dynamics(t, y, u_net, gamma):
    """
    Augmented dynamics function that includes cumulative cost.

    y: [2] tensor where y[0] = x, y[1] = c
    """
    # Ensure y is a 1D tensor with 2 elements
    if y.dim() != 1 or y.shape[0] != 2:
        raise ValueError(f"Expected y to be a 1D tensor with 2 elements, got shape {y.shape}")

    # Extract x and c from y
    x = y[0].unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]
    c = y[1]  # Scalar

    # Prepare t_tensor to match batch size of x
    t_tensor = t * torch.ones_like(x)  # Shape: [1, 1]

    # Compute control u(t, x)
    u = u_net(t_tensor, x)  # Shape: [1, 1]

    # Compute dynamics
    dxdt = gamma * u * x  # Shape: [1, 1]
    dcdt = ((1 - u) * x).squeeze(1)  # Shape: [1]

    # Concatenate dxdt and dcdt to form dy_dt
    dy_dt = torch.cat([dxdt.squeeze(1), dcdt], dim=0)  # Shape: [2]

    return dy_dt

def compute_optimal_control(t, gamma, T):
    """
    Computes the analytical optimal control u*(t).

    u*(t) = 1 for t < T - 1/gamma
    u*(t) = 0 otherwise

    If T - 1/gamma < 0, then u*(t) = 0 for all t.

    Parameters:
    t (torch.Tensor): Time tensor.
    gamma (float): Parameter in the dynamics.
    T (float): Terminal time.

    Returns:
    torch.Tensor: Optimal control values at time t.
    """
    switching_time = T - (1 / gamma)
    if switching_time < 0:
        # No switching within [0, T], u*(t) = 0 for all t
        u_star = torch.zeros_like(t)
    else:
        u_star = torch.where(t < switching_time, torch.ones_like(t), torch.zeros_like(t))
    return u_star

# Function to load models
def load_models(model_dir, state_dim):
    u_net = UNet(state_dim=state_dim)
    j_net = JNet(state_dim=state_dim)
    # Load the models from the last stage (stage 0)
    u_net_path = os.path.join(model_dir, 'u_net_stage_0.pth')
    j_net_path = os.path.join(model_dir, 'j_net_stage_0.pth')
    if not os.path.exists(u_net_path) or not os.path.exists(j_net_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}. Please ensure the models are trained and saved correctly.")
    # Load models on CPU
    u_net.load_state_dict(torch.load(u_net_path, map_location=torch.device('cpu')))
    j_net.load_state_dict(torch.load(j_net_path, map_location=torch.device('cpu')))
    return u_net, j_net


if __name__ == '__main__':
    main()