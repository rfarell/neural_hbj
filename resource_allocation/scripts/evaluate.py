# evaluate.py

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import UNet, JNet, compute_ground_truth, load_models

def plot_predictions(u_star, u_pred, J_star, J_pred):
    # Plotting error distributions
    os.makedirs('../results/resource_allocation/plots', exist_ok=True)

    u_error = (u_pred - u_star).squeeze().numpy()
    plt.figure()
    plt.hist(u_error, bins=50)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'U-net Error Distribution')
    plt.savefig(f'../results/resource_allocation/plots/u_net_error_distribution.png')
    plt.close()

    J_error = (J_pred - J_star).squeeze().numpy()
    plt.figure()
    plt.hist(J_error, bins=50)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'J-net Error Distribution')
    plt.savefig(f'../results/resource_allocation/plots/j_net_error_distribution.png')
    plt.close()


def main():
    gamma = 0.5
    T = 4.0
    state_dim = 1
    num_test_samples = 1024
    model_dir = '../models/resource_allocation'

    mse_results = []

    print('\nEvaluating Resource Allocation Problem')
    # Load the trained models
    u_net_model, j_net_model = load_models(model_dir, state_dim)

    # Generate test data
    t_test = torch.rand(num_test_samples, 1) * T
    x_test = torch.rand(num_test_samples, state_dim) * 4 + 1  # x in [1,5]

    # Compute ground truth
    u_star  = compute_ground_truth(t_test, T, gamma)

    # Compute predictions
    with torch.no_grad():
        u_pred = u_net_model(t_test, x_test)

    # Compute MSE
    mse_u = torch.mean((u_pred - u_star) ** 2).item()

    mse_results.append({'MSE_U': mse_u})

    # Save MSE results to a CSV file
    mse_df = pd.DataFrame(mse_results)
    os.makedirs('../results/resource_allocation/mse_tables', exist_ok=True)
    mse_df.to_csv('../results/resource_allocation/mse_tables/mse_results.csv', index=False)

if __name__ == '__main__':
    main()

