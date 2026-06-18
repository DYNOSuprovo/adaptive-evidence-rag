#!/bin/bash
# Setup environment for Adaptive Evidence-Aware RAG

set -e

echo "========================================"
echo "Environment Setup Script"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA if available
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected! Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing requirements..."
pip install -r ../requirements.txt

echo ""
echo "========================================"
echo "Setup complete!"
echo "Activate environment: source venv/bin/activate"
echo "Run notebook: jupyter notebook notebooks/train_evidence_rag.ipynb"
echo "========================================"
