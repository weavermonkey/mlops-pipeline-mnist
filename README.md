# MLOps Pipeline MNIST

[![Build Status](https://github.com/weavermonkey/mlops-pipeline-mnist/actions/workflows/test.yml/badge.svg)](https://github.com/weavermonkey/mlops-pipeline-mnist/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/weavermonkey/mlops-pipeline-mnist)
![Code Size](https://img.shields.io/github/languages/code-size/weavermonkey/mlops-pipeline-mnist)

### Model Metrics
![Accuracy](https://img.shields.io/badge/accuracy-95%25-success)
![Parameters](https://img.shields.io/badge/parameters-<25K-informational)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A lightweight CNN model for MNIST digit classification with MLOps practices.

## Model Architecture

```
Input: 28×28×1 grayscale images
```

| Layer | Type | Input Shape | Kernel | Output Shape | Activation | Pooling |
|-------|------|-------------|--------|--------------|------------|---------|
| **Conv1** | Conv2d | 28×28×1 | 3×3, 12 filters, padding=1 | 28×28×12 | LeakyReLU(0.1) | MaxPool2d(2) |
| | | | | **→ 14×14×12** | | |
| **Conv2** | Conv2d | 14×14×12 | 3×3, 16 filters, padding=1 | 14×14×16 | LeakyReLU(0.1) | MaxPool2d(2) |
| | | | | **→ 7×7×16** | | |
| **Conv3** | Conv2d | 7×7×16 | 3×3, 20 filters, padding=1 | 7×7×20 | LeakyReLU(0.1) | MaxPool2d(2) |
| | | | | **→ 3×3×20** | | |
| **FC1** | Linear | 180 (flattened) | - | 32 | LeakyReLU(0.1) | - |
| **FC2** | Linear | 32 | - | 10 | None (logits) | - |

### Architecture Flow
```
28×28×1 → [Conv1+BN+ReLU+Pool] → 14×14×12 → [Conv2+BN+ReLU+Pool] → 7×7×16 → [Conv3+BN+ReLU+Pool] → 3×3×20 → [Dropout] → [FC1+ReLU] → 32 → [FC2] → 10
```

## Model Specifications

- **Total Parameters**: 10,982
- **First Epoch Accuracy**: >95% (without data augmentation)
- **Batch Size**: 32
- **Learning Rate**: 0.003
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Regularization

- Batch Normalization after each conv layer
- Dropout(0.1) before FC1
- **Data augmentation is optional** (disabled by default):
  - Rotation: ±5 degrees
  - Translation: ±5% in any direction
  - Scaling: 95-105% of original size
  - Applied with 20% probability when enabled

## Data Augmentation
The model includes data augmentation to improve robustness. Each image has a 20% chance of being augmented using one of these transformations:

| Transformation | Parameters |
|---------------|------------|
| Rotation | ±5 degrees |
| Translation | ±5% in any direction |
| Scaling | 95-105% of original size |

### Augmentation Examples
Below are examples of different augmentation techniques applied to MNIST digits:

![Augmentation Examples](augmentation_samples.png)

Each row shows an original digit followed by three augmentation techniques:
- **Rotation**: Applied 10° rotation
- **Scale**: Reduced to 90% of original size
- **Translation**: Shifted 10% to the right

## Training

```python
python model.py
```

## Testing

```python
pytest test_model.py
```

Tests include:
- Parameter count validation
- First epoch accuracy check
- Gradient flow verification
- Noise robustness testing
- Augmentation validation

## GitHub Actions
The pipeline automatically tests:
1. Model parameter count (must be < 25,000)
2. First epoch accuracy (must be > 95%)
3. Gradient flow through all layers
4. Model robustness to noise
5. Augmentation effectiveness
