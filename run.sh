#!/bin/bash
# Run all experiments for Neural HJB

echo "🚀 Running Neural HJB Experiments"
echo "================================="

# Check if conda environment exists
if ! conda env list | grep -q "neural_hjb"; then
    echo "⚠️  Conda environment 'neural_hjb' not found."
    echo "Please run ./setup.sh first to create the environment."
    exit 1
fi

# Function to run commands in conda environment
run_in_env() {
    conda run -n neural_hjb "$@"
}

# Motion Control Experiments
echo ""
echo "📊 Motion Control Experiments"
echo "----------------------------"
cd motion_control/scripts

echo "Training models for dimensions 2, 4, 6, 8..."
run_in_env python train.py --dimensions 2 4 6 8 --num_epochs 1000

echo "Evaluating models..."
run_in_env python evaluate.py

echo "Generating heatmaps..."
run_in_env python heatmaps.py

cd ../..

# Resource Allocation Experiments
echo ""
echo "📊 Resource Allocation Experiments"
echo "---------------------------------"
cd resource_allocation/scripts

echo "Training resource allocation model..."
run_in_env python train.py --num_epochs 1000

echo "Evaluating model..."
run_in_env python evaluate.py

echo "Showing various control paths..."
run_in_env python show_various_paths.py

cd ../..

echo ""
echo "✅ All experiments complete!"
echo ""
echo "Results can be found in:"
echo "  - motion_control/results/"
echo "  - resource_allocation/results/"