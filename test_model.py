import pytest
import torch
import torch.nn.functional as F
from model import LightweightCNN, train_epoch, load_data, AugmentedMNIST

def test_parameter_count():
    model = LightweightCNN()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"
    print(f"Parameter count: {param_count}")

def test_first_epoch_accuracy():
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    train_loader = load_data()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    torch.manual_seed(42)  # For reproducibility
    accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    
    assert accuracy >= 95.0, f"First epoch accuracy is {accuracy}%, should be at least 95%"
    print(f"First epoch accuracy: {accuracy}%")

def test_gradient_flow():
    """Test if gradients are properly flowing through all layers of the model"""
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    model.train()
    
    # Create sample input
    x = torch.randn(1, 1, 28, 28).to(device)
    target = torch.tensor([5]).to(device)  # Random target class
    
    # Forward pass
    output = model(x)
    loss = F.cross_entropy(output, target)
    loss.backward()
    
    # Check if gradients exist and are non-zero for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.any(param.grad != 0), f"Gradient for {name} is all zeros"
        print(f"Gradient check passed for {name}")

def test_gradient_magnitudes():
    """Test for exploding or vanishing gradients"""
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    model.train()
    
    train_loader = load_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # Get first batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Forward and backward pass
    optimizer.zero_grad()
    output = model(images)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    
    # Check gradient magnitudes
    for name, param in model.named_parameters():
        grad_norm = torch.norm(param.grad.data)
        
        # Different thresholds for different parameter types
        if 'weight' in name and 'conv' in name:
            max_threshold = 2.0
            min_threshold = 1e-6
        elif 'weight' in name and 'fc' in name:
            max_threshold = 2.0
            min_threshold = 1e-6
        elif 'bias' in name and 'conv' in name:
            max_threshold = 1.0
            min_threshold = 1e-8
        elif 'bias' in name:
            max_threshold = 1.0
            min_threshold = 1e-7
        else:  # BatchNorm and other parameters
            max_threshold = 1.0
            min_threshold = 1e-6
            
        assert grad_norm < max_threshold, f"Exploding gradient in {name}: {grad_norm}"
        assert grad_norm > min_threshold, f"Vanishing gradient in {name}: {grad_norm}"
        print(f"Gradient magnitude check passed for {name}: {grad_norm:.6f}")

def test_noisy_inputs():
    """Test model's robustness to input noise"""
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    model.eval()
    
    train_loader = load_data()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    torch.manual_seed(42)
    
    # Train for one epoch to get a decent model
    train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Get a batch for testing
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        clean_output = model(images)
        clean_pred = torch.argmax(clean_output, dim=1)
    
        # Test different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        min_accuracies = [0.8, 0.7, 0.6]
        
        torch.manual_seed(42)  # For reproducible noise
        for noise_level, min_accuracy in zip(noise_levels, min_accuracies):
            # Add Gaussian noise
            noise = torch.randn_like(images) * noise_level
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0, 1)
            
            # Get predictions on noisy images
            noisy_output = model(noisy_images)
            noisy_pred = torch.argmax(noisy_output, dim=1)
            
            # Calculate accuracy
            accuracy = (noisy_pred == labels).float().mean().item()
            assert accuracy >= min_accuracy, f"Model performs poorly with noise level {noise_level}: {accuracy:.2f} accuracy"
            print(f"Noise robustness test passed for noise level {noise_level}: {accuracy:.2f} accuracy")

def test_augmentation():
    """Test if augmentation produces valid images and maintains original label"""
    dataset = AugmentedMNIST(train=True)
    
    # Test a few random samples
    for _ in range(5):
        idx = torch.randint(0, len(dataset), (1,)).item()
        original_img, original_label = dataset.original_dataset[idx]
        augmented_img, augmented_label = dataset[idx]
        
        # Check if label is preserved
        assert original_label == augmented_label, "Augmentation changed the label"
        
        # Check if image is valid
        assert torch.is_tensor(augmented_img), "Augmented image is not a tensor"
        assert augmented_img.shape == (1, 28, 28), f"Wrong shape: {augmented_img.shape}"
        assert augmented_img.min() >= 0 and augmented_img.max() <= 1, "Image values out of range [0,1]"
        
    print("Augmentation test passed")

def test_augmentation_comparison():
    """Compare model performance with and without augmentation"""
    device = torch.device("cpu")
    criterion = torch.nn.CrossEntropyLoss()
    torch.manual_seed(42)  # For reproducibility
    
    # Test without augmentation
    model_standard = LightweightCNN().to(device)
    train_loader_standard = load_data(use_augmentation=False)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=0.003)
    accuracy_standard = train_epoch(model_standard, train_loader_standard, criterion, optimizer_standard, device)
    
    # Test with augmentation
    model_augmented = LightweightCNN().to(device)
    train_loader_augmented = load_data(use_augmentation=True)
    optimizer_augmented = torch.optim.Adam(model_augmented.parameters(), lr=0.003)
    accuracy_augmented = train_epoch(model_augmented, train_loader_augmented, criterion, optimizer_augmented, device)
    
    print("\nAccuracy Comparison:")
    print(f"Standard training: {accuracy_standard:.2f}%")
    print(f"Augmented training: {accuracy_augmented:.2f}%")
    
    # Both should meet minimum accuracy requirement
    assert accuracy_standard >= 95.0, f"Standard training accuracy {accuracy_standard:.2f}% below 95%"
    assert accuracy_augmented >= 95.0, f"Augmented training accuracy {accuracy_augmented:.2f}% below 95%"
    
    print("Both models meet accuracy requirements")

if __name__ == "__main__":
    pytest.main([__file__])