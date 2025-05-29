"""Training script for motion control HJB solver."""

import argparse
import os
import sys
from typing import List

import torch
import torch.optim as optim

from utils import UNet, JNet, TrainStage, save_models


def main() -> None:
    parser = argparse.ArgumentParser(description='Train U-net and J-net for Motion Control Problem')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[2, 4, 6, 8],
                        help='List of dimensions to train on (e.g., --dimensions 2 4 6 8)')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs per stage')
    args = parser.parse_args()

    # Validate arguments
    if args.num_epochs <= 0:
        print("Error: Number of epochs must be positive")
        sys.exit(1)
    
    for dim in args.dimensions:
        if dim <= 0:
            print(f"Error: Dimension {dim} must be positive")
            sys.exit(1)

    # Define total time horizon and stages
    T = 1.0
    K = 5  # Number of stages
    T_stages = torch.linspace(0, T, steps=K+1)

    for n in args.dimensions:
        print(f'\nTraining for dimension n={n}')
        state_dim = n
        output_dir = f'../models/n{n}'
        os.makedirs(output_dir, exist_ok=True)

        # Initialize networks
        u_net_model = UNet(state_dim=state_dim)
        j_net_model = JNet(state_dim=state_dim)

        # Define optimizer
        optimizer = optim.Adam(list(u_net_model.parameters()) + list(j_net_model.parameters()), lr=1e-3)

        # Create lists to store losses
        training_losses = []

        try:
            # Staged training procedure
            for k in range(K-1, -1, -1):
                T_start = T_stages[k].item()
                T_end = T
                print(f'\nTraining on stage {k}, time interval [{T_start}, {T_end}]')
                stage_losses = TrainStage(T_start, T_end, u_net_model, j_net_model, optimizer,
                                          num_epochs=args.num_epochs, state_dim=state_dim)
                training_losses.extend(stage_losses)

                # Save model checkpoints
                save_models(u_net_model, j_net_model, output_dir, stage=k)

            # Save training losses for plotting
            os.makedirs('../results/data', exist_ok=True)
            torch.save(training_losses, f'../results/data/n{n}_training_losses.pt')
            print(f'Successfully completed training for dimension n={n}')
            
        except Exception as e:
            print(f"Error during training for dimension n={n}: {str(e)}")
            continue

if __name__ == '__main__':
    main()
