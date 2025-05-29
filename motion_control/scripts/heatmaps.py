# heatmaps.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import UNet, JNet, compute_ground_truth, load_models

def main():
    # Set parameters
    n = 2  # Dimension
    T = 1.0
    state_dim = n
    model_dir = f'../models/n{n}'
    
    # Load the trained models
    u_net_model, j_net_model = load_models(model_dir, state_dim)
    
    # Set time values
    t_values = [0.0, 0.33, 0.67, 1.0]
    
    # Create a grid over the 2D space
    grid_size = 100  # Adjust for resolution
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Flatten the grid for batch processing
    x_flat = X.flatten()
    y_flat = Y.flatten()
    xy_grid = np.stack([x_flat, y_flat], axis=1)
    
    # Convert grid to torch tensors
    x_grid = torch.tensor(xy_grid, dtype=torch.float32)
    
    # Prepare figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for idx, t in enumerate(t_values):
        print(f'Processing time t = {t}')
        t_tensor = torch.ones(x_grid.shape[0], 1) * t  # Shape: [num_points, 1]
        
        # Compute ground truth
        u_star, J_star = compute_ground_truth(t_tensor, x_grid, T)
        
        # Compute model predictions
        with torch.no_grad():
            u_pred = u_net_model(t_tensor, x_grid)
            J_pred = j_net_model(t_tensor, x_grid)
        
        # Compute errors
        J_error = (J_pred - J_star).squeeze().numpy()
        u_error = (u_pred - u_star).norm(dim=1).numpy()
        
        # Reshape errors to grid shape
        J_error_grid = J_error.reshape(grid_size, grid_size)
        u_error_grid = u_error.reshape(grid_size, grid_size)
        
        # Reshape J_pred and J_star for plotting
        J_pred_grid = J_pred.squeeze().numpy().reshape(grid_size, grid_size)
        J_star_grid = J_star.squeeze().numpy().reshape(grid_size, grid_size)
        
        # Reshape u vectors for quiver plot
        u_pred_np = u_pred.numpy()
        U = u_pred_np[:, 0].reshape(grid_size, grid_size)
        V = u_pred_np[:, 1].reshape(grid_size, grid_size)
        
        # First row: Heatmap of J_pred vs ground truth J_star
        ax1 = axes[0, idx]
        im1 = ax1.imshow(J_pred_grid - J_star_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='RdBu')
        ax1.set_title(f'J Error at t={t}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        fig.colorbar(im1, ax=ax1)
        
        # Second row: Heatmap of magnitude error of U-net vs ground truth
        ax2 = axes[1, idx]
        im2 = ax2.imshow(u_error_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
        ax2.set_title(f'U Magnitude Error at t={t}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        fig.colorbar(im2, ax=ax2)
        
        # Third row: Quiver plot of U vectors
        ax3 = axes[2, idx]
        skip = (slice(None, None, 5), slice(None, None, 5))  # Adjust for quiver density
        ax3.quiver(X[skip], Y[skip], U[skip], V[skip])
        ax3.set_title(f'U Vectors at t={t}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    os.makedirs('../results/plots/heatmaps', exist_ok=True)
    plt.savefig('../results/plots/heatmaps/heatmaps_n2.png')
    plt.show()

if __name__ == '__main__':
    main()
