import pytest
import torch
import torch.nn.functional as F
from model import LightweightCNN, train_epoch, load_data

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
    
    # Check gradient magnitudes with different thresholds for weights and biases
    for name, param in model.named_parameters():
        grad_norm = torch.norm(param.grad.data)
        
        # Different thresholds for different parameter types
        if 'weight' in name and 'conv' in name:
            max_threshold = 2.0  # Higher threshold for conv weights
            min_threshold = 1e-6
        elif 'weight' in name and 'fc' in name:
            max_threshold = 2.0  # Higher threshold for FC weights
            min_threshold = 1e-6
        elif 'bias' in name and 'conv' in name:
            max_threshold = 1.0
            min_threshold = 1e-8  # Much lower threshold for conv biases
        elif 'bias' in name:
            max_threshold = 1.0
            min_threshold = 1e-7  # Original bias threshold
        else:  # BatchNorm and other parameters
            max_threshold = 1.0
            min_threshold = 1e-6
            
        assert grad_norm < max_threshold, f"Exploding gradient in {name}: {grad_norm}"
        assert grad_norm > min_threshold, f"Vanishing gradient in {name}: {grad_norm}"
        print(f"Gradient magnitude check passed for {name}: {grad_norm:.6f}")

if __name__ == "__main__":
    pytest.main([__file__])