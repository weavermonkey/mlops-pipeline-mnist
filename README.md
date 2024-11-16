# MLOps Pipeline MNIST

A lightweight CNN model for MNIST digit classification with MLOps practices.

## Model Specifications
- Parameters: < 25,000
- First Epoch Accuracy: > 95%

## Requirements
- Python 3.10
- PyTorch
- torchvision
- pytest

## Running Tests Locally
```bash
# Install requirements
pip install -r requirements.txt

# Run tests
pytest test_model.py -v
```

## GitHub Actions
The pipeline automatically tests:
1. Model parameter count (must be < 25,000)
2. First epoch accuracy (must be > 95%)
