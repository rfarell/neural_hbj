# evaluate.py

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import UNet, JNet, compute_ground_truth, load_models

def main():
    dimensions = [2, 4, 6, 8]
    T = 1.0
    num_test_samples = 1024

    mse_results = []

    for n in dimensions:
        print(f'\nEvaluating for dimension n={n}')
        state_dim = n
        model_dir = f'../models/n{n}'

        # Load the trained models
        u_net_model, j_net_model = load_models(model_dir, state_dim)

        # Generate test data
        t_test = torch.rand(num_test_samples, 1) * T
        x_test = torch.rand(num_test_samples, state_dim) * 4 - 2  # x in [-2,2]^n

        # Compute ground truth
        u_star, J_star = compute_ground_truth(t_test, x_test, T)

        # Compute predictions
        with torch.no_grad():
            u_pred = u_net_model(t_test, x_test)
            J_pred = j_net_model(t_test, x_test)

        # Compute MSE
        mse_u = torch.mean((u_pred - u_star) ** 2).item()
        mse_J = torch.mean((J_pred - J_star) ** 2).item()

        mse_results.append({'Dimension': n, 'MSE_U': mse_u, 'MSE_J': mse_J})

        # Save plots of u_pred vs u_star and J_pred vs J_star
        plot_predictions(u_star, u_pred, J_star, J_pred, n)

    # Save MSE results to a CSV file
    mse_df = pd.DataFrame(mse_results)
    os.makedirs('../results/mse_tables', exist_ok=True)
    mse_df.to_csv('../results/mse_tables/mse_results.csv', index=False)

    # Generate plots of MSE vs Dimension
    plt.figure()
    plt.plot(mse_df['Dimension'], mse_df['MSE_U'], marker='o', label='MSE_U')
    plt.plot(mse_df['Dimension'], mse_df['MSE_J'], marker='s', label='MSE_J')
    plt.xlabel('Dimension')
    plt.ylabel('MSE')
    plt.title('MSE vs Dimension')
    plt.legend()
    os.makedirs('../results/plots', exist_ok=True)
    plt.savefig('../results/plots/mse_vs_dimension.png')
    plt.close()

    # Plot training losses
    plot_training_losses(dimensions)

def plot_predictions(u_star, u_pred, J_star, J_pred, n):
    # Plotting function to compare predictions and ground truth
    # For high dimensions, you might plot histograms or error distributions
    os.makedirs(f'../results/plots/n{n}', exist_ok=True)
    # Example: Plotting histograms of errors
    u_error = (u_pred - u_star).norm(dim=1).numpy()
    plt.figure()
    plt.hist(u_error, bins=50)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'U-net Error Distribution for n={n}')
    plt.savefig(f'../results/plots/n{n}/u_net_error_distribution.png')
    plt.close()

    J_error = (J_pred - J_star).squeeze().numpy()
    plt.figure()
    plt.hist(J_error, bins=50)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'J-net Error Distribution for n={n}')
    plt.savefig(f'../results/plots/n{n}/j_net_error_distribution.png')
    plt.close()

def plot_training_losses(dimensions):
    for n in dimensions:
        training_losses_path = f'../results/data/n{n}_training_losses.pt'
        if os.path.exists(training_losses_path):
            training_losses = torch.load(training_losses_path)
            plt.figure()
            plt.plot(training_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss Curve for n={n}')
            os.makedirs(f'../results/plots/training_curves', exist_ok=True)
            plt.savefig(f'../results/plots/training_curves/n{n}_training_loss.png')
            plt.close()
        else:
            print(f'Training losses for n={n} not found.')

if __name__ == '__main__':
    main()
