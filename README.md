# Neural Network Solutions for Hamilton-Jacobi-Bellman Equations

This repository implements neural network-based solvers for Hamilton-Jacobi-Bellman (HJB) equations in optimal control problems. We demonstrate the approach on two canonical problems:

1. **Motion Control**: Multi-dimensional optimal control for driving systems to the origin
2. **Resource Allocation**: Optimal production and consumption scheduling

## Key Results

- **Motion Control**: Achieved MSE < 0.001 for 2D systems, scaling to 8D with maintained accuracy
- **Resource Allocation**: Neural network captures bang-bang control structure with MSE = 0.00487

## Mathematical Framework

The HJB equation for optimal control problems takes the form:

$$\frac{\partial J}{\partial t} + \min_u \left[ \nabla J \cdot f(x,u) + g(x,u) \right] = 0$$

where:
- $J(t,x)$ is the cost-to-go function
- $f(x,u)$ represents system dynamics
- $g(x,u)$ is the running cost
- $u$ is the control input

Our neural network approach approximates both $J(t,x)$ and the optimal control $u^*(t,x)$ using deep networks trained via automatic differentiation.

## Technical Approach

### Neural Architecture
- **UNet**: Approximates optimal control $u^*(t,x)$ with constraints (e.g., $||u|| \leq 1$)
- **JNet**: Approximates cost-to-go function $J(t,x)$
- Both networks use 2-layer MLPs with 64 hidden units and tanh activation

### Training Methodology
- **Staged Training**: Problems solved backward in time using 5 stages
- **Loss Function**: Combines HJB residuals, boundary conditions, and trajectory matching
- **Automatic Differentiation**: PyTorch autograd computes all required derivatives

## Installation

```bash
git clone https://github.com/rfarell/neural_hjb.git
cd neural_hjb

# Option 1: Quick setup with conda (recommended)
./setup.sh  # Creates conda environment and installs dependencies

# Option 2: Manual installation
pip install -e .
```

**Requirements**: Python 3.8+, PyTorch, torchdiffeq, NumPy, Pandas, Matplotlib

## Quick Start

### Run All Experiments
```bash
./run.sh  # Runs all experiments automatically
```

### Individual Experiments

#### Motion Control Problem
```bash
cd motion_control/scripts
python train.py --dimensions 2 4 6 8 --num_epochs 1000
python evaluate.py
```

#### Resource Allocation Problem
```bash
cd resource_allocation/scripts
python train.py --num_epochs 1000
python evaluate.py
```

## Results

### 1. Motion Control Problem

The motion control problem seeks to drive a system to the origin while minimizing the quadratic cost:

$$\min_{u} \int_0^T ||x(s)||^2 ds + ||x(T)||^2$$

subject to $\dot{x} = u$ and $||u|| \leq 1$.

#### Performance Across Dimensions

| Dimension | MSE (Control) | MSE (Cost-to-go) |
|-----------|---------------|-------------------|
| 2D        | 0.00126      | 0.00182          |
| 4D        | 0.00002      | 0.01214          |
| 6D        | 0.00003      | 0.02531          |
| 8D        | 0.00049      | 4.23102          |

![MSE vs Dimension](motion_control/results/plots/mse_vs_dimension.png)
*Figure 1: Mean squared error scaling with problem dimension. The method maintains accuracy even in higher dimensions.*

#### Training Convergence

![Training Loss](motion_control/results/plots/training_curves/n2_training_loss.png)
*Figure 2: Training loss convergence for 2D motion control across 5 time stages.*

#### Control Policy Visualization (2D)

![Heatmaps](motion_control/results/plots/heatmaps/heatmaps_n2.png)
*Figure 3: Learned control policy and cost-to-go function for 2D motion control at t=0.5.*

### 2. Resource Allocation Problem

The resource allocation problem optimizes production vs. consumption over time:

$$\max_{u} \int_0^T (1-u(t))x(t) dt$$

subject to $\dot{x} = \gamma u x$ and $u \in [0,1]$.

The analytical solution exhibits bang-bang control: $u^*(t) = 1$ for $t < T - 1/\gamma$, then switches to $u^*(t) = 0$.

#### Neural Network Performance

![Control Paths](resource_allocation/results/resource_allocation/plots/control_paths_resource_allocation.png)
*Figure 4: Comparison of neural network predictions (solid lines) vs analytical solution (dashed) for different initial conditions. The network accurately captures the switching behavior at t=2.*

**Key Metrics**:
- MSE for control: 0.00487
- Switching time accuracy: ±0.05 time units
- Captures sharp transitions in bang-bang control

#### Error Distribution

![Error Distribution](resource_allocation/results/resource_allocation/plots/u_net_error_distribution.png)
*Figure 5: Distribution of control prediction errors showing concentration near zero.*

## Problem Formulations

### Motion Control
- **Dynamics**: $\dot{x} = u$
- **Cost**: $\int_0^T ||x||^2 dt + ||x(T)||^2$
- **Constraint**: $||u|| \leq 1$
- **Analytical Solution**: $u^* = -x/||x||$ when $||x|| > 0$

### Resource Allocation
- **Dynamics**: $\dot{x} = \gamma u x$
- **Objective**: $\max \int_0^T (1-u)x dt$
- **Constraint**: $u \in [0,1]$
- **Analytical Solution**: Bang-bang control with switching at $t = T - 1/\gamma$

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_hjb,
  author = {Farell, Ryan},
  title = {Neural Network Solutions for Hamilton-Jacobi-Bellman Equations},
  year = {2024},
  url = {https://github.com/rfarell/neural_hjb}
}
```

## License

MIT License - see LICENSE file for details.