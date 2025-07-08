#!/bin/bash

# Install dependencies with conflict resolution
# This script handles dependency conflicts and provides fallback options

set -e

echo "🔧 Installing dependencies with conflict resolution..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Consider creating one:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate  # On Linux/Mac"
    echo "   venv\\Scripts\\activate     # On Windows"
    echo ""
fi

# Try to install with development requirements first
echo "📦 Attempting to install with flexible version constraints..."
if pip install -r requirements-dev.txt; then
    echo "✅ Successfully installed dependencies with flexible constraints"
    exit 0
fi

echo "⚠️  Flexible installation failed, trying with exact versions..."

# If that fails, try with exact versions
if pip install -r requirements.txt; then
    echo "✅ Successfully installed dependencies with exact versions"
    exit 0
fi

echo "❌ Both installation methods failed. Attempting manual conflict resolution..."

# Manual conflict resolution
echo "🔧 Installing core dependencies first..."
pip install tensorflow==2.13.0
pip install azureml-core==1.53.0
pip install azureml-pipeline==1.53.0
pip install azureml-pipeline-steps==1.53.0
pip install azureml-train==1.53.0
pip install azureml-train-automl==1.53.0
pip install azureml-mlflow==1.53.0
pip install mlflow==2.6.0

echo "🔧 Installing Azure storage with compatible version..."
pip install azure-storage-blob==12.13.0

echo "🔧 Installing remaining dependencies..."
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install opencv-python==4.8.0.76
pip install fastapi==0.103.1
pip install uvicorn==0.23.2
pip install pydantic==2.0.3

echo "🔧 Installing monitoring and security tools..."
pip install applicationinsights==0.11.10
pip install opencensus==0.11.0
pip install opencensus-ext-azure==1.1.8
pip install bandit==1.7.5
pip install safety==2.3.5
pip install semgrep==1.40.0

echo "🔧 Installing testing tools..."
pip install pytest==7.4.2
pip install pytest-cov==4.1.0
pip install pytest-asyncio==0.21.1

echo "🔧 Installing infrastructure tools..."
pip install kubernetes==27.2.0
pip install azure-identity==1.13.0
pip install azure-keyvault-secrets==4.6.0

echo "🔧 Installing utilities..."
pip install python-dotenv==1.0.0
pip install pyyaml==6.0.1
pip install click==8.1.7
pip install rich==13.5.2
pip install tqdm==4.66.1

echo "🔧 Installing development tools..."
pip install black==23.7.0
pip install flake8==6.0.0
pip install mypy==1.5.1
pip install pre-commit==3.4.0

echo "✅ All dependencies installed successfully!"

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import tensorflow as tf
import azureml.core
import fastapi
import kubernetes
print('✅ Core dependencies verified successfully!')
print(f'TensorFlow version: {tf.__version__}')
print(f'Azure ML version: {azureml.core.__version__}')
print(f'FastAPI version: {fastapi.__version__}')
" 