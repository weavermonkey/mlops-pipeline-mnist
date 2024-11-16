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

if __name__ == "__main__":
    pytest.main([__file__])