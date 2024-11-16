import pytest
import torch
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

if __name__ == "__main__":
    pytest.main([__file__])
