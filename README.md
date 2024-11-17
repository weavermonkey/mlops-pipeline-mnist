# MLOps Pipeline MNIST

A lightweight CNN model for MNIST digit classification with MLOps practices.

## Model Specifications
- Parameters: < 25,000
- First Epoch Accuracy: > 95%

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

## GitHub Actions
The pipeline automatically tests:
1. Model parameter count (must be < 25,000)
2. First epoch accuracy (must be > 95%)
3. Gradient flow through all layers
4. Model robustness to noise
5. Augmentation effectiveness