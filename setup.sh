#!/bin/bash
PROJECT_NAME=${PWD##*/}
echo "🚀 Setting up $PROJECT_NAME environment..."

# Create & activate conda env
conda create -n $PROJECT_NAME python=3.11 -y
echo "✅ Created conda env: $PROJECT_NAME"

# Install dependencies
echo "📦 Installing dependencies..."
conda run -n $PROJECT_NAME pip install -r requirements.txt 2>/dev/null || echo "⚠️  No requirements.txt found"

echo "
✨ Setup complete! Next steps:
1. Activate env: conda activate $PROJECT_NAME
2. Run experiments: ./run.sh
"