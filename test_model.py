import pytest
import torch
import torch.nn.functional as F
from model import LightweightCNN, train_epoch, load_data
import numpy as np

# Original tests
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

# New tests for model robustness and stability
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
    
    # Training loop with gradient checking
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
        assert grad_norm < 10, f"Exploding gradient in {name}: {grad_norm}"
        assert grad_norm > 1e-5, f"Vanishing gradient in {name}: {grad_norm}"
        print(f"Gradient magnitude check passed for {name}: {grad_norm:.6f}")

def test_noisy_inputs():
    """Test model's robustness to input noise"""
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    model.eval()
    
    # Get clean test data
    train_loader = load_data()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions for clean images
    with torch.no_grad():
        clean_output = model(images)
        clean_pred = torch.argmax(clean_output, dim=1)
    
    # Add Gaussian noise
    noise_levels = [0.1, 0.2, 0.3]
    for noise_level in noise_levels:
        noisy_images = images + noise_level * torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, 0, 1)  # Ensure valid pixel values
        
        with torch.no_grad():
            noisy_output = model(noisy_images)
            noisy_pred = torch.argmax(noisy_output, dim=1)
        
        # Check if accuracy drop is within acceptable range
        accuracy = (noisy_pred == labels).float().mean().item()
        assert accuracy > 0.7, f"Model performs poorly with noise level {noise_level}: {accuracy:.2f} accuracy"
        print(f"Noise robustness test passed for noise level {noise_level}: {accuracy:.2f} accuracy")

def test_edge_cases():
    """Test model with edge case inputs"""
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    model.eval()
    
    test_cases = [
        ("zeros", torch.zeros(1, 1, 28, 28)),
        ("ones", torch.ones(1, 1, 28, 28)),
        ("dim", torch.rand(1, 1, 28, 28) * 0.1),
        ("bright", torch.rand(1, 1, 28, 28) * 0.9)
    ]
    
    for name, test_input in test_cases:
        test_input = test_input.to(device)
        with torch.no_grad():
            output = model(test_input)
            
        # Check if output is valid
        assert not torch.isnan(output).any(), f"Model produced NaN outputs for {name} case"
        assert not torch.isinf(output).any(), f"Model produced infinite outputs for {name} case"
        
        # Check if probabilities sum to approximately 1
        probs = F.softmax(output, dim=1)
        assert torch.abs(probs.sum() - 1) < 1e-6, f"Output probabilities don't sum to 1 for {name} case"
        print(f"Edge case test passed for {name} input")

def test_batch_sizes():
    """Test model with different batch sizes"""
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    model.eval()
    
    # Test various batch sizes
    batch_sizes = [1, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        # Create random input of specified batch size
        x = torch.randn(batch_size, 1, 28, 28).to(device)
        
        try:
            with torch.no_grad():
                output = model(x)
            
            # Check output shape
            assert output.shape == (batch_size, 10), f"Wrong output shape for batch size {batch_size}"
            
            # Check if outputs are valid
            assert not torch.isnan(output).any(), f"NaN outputs for batch size {batch_size}"
            assert not torch.isinf(output).any(), f"Infinite outputs for batch size {batch_size}"
            print(f"Batch size test passed for size {batch_size}")
            
        except Exception as e:
            pytest.fail(f"Failed to process batch size {batch_size}: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])