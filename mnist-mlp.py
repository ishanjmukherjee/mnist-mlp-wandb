# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
import os
from torch.utils.data import DataLoader

# Initialize wandb
wandb.login()
wandb.init(project="mnist-mlp")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%

# Define hyperparameters
config = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 10,
    "hidden_size": 512
}

# Log hyperparameters
wandb.config.update(config)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, hidden_size=512):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.layers(x)

# %%

# Load and transform data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download datasets to a specific location (useful for remote servers)
data_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(data_dir, exist_ok=True)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# %%

# Initialize model
model = MLP(hidden_size=config["hidden_size"]).to(device)
wandb.watch(model, log="all")  # Log gradients and parameters

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# %%

# Training loop
def train():
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log batch statistics
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        # Log epoch statistics
        train_accuracy = 100. * correct / total
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_accuracy
        })

        # Evaluate on test set
        test()

        # Save model checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
        wandb.save(f"model_epoch_{epoch}.pt")

# Evaluation function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # Log test statistics
    test_accuracy = 100. * correct / total
    wandb.log({
        "test_loss": test_loss / len(test_loader),
        "test_accuracy": test_accuracy
    })

    print(f'Test Loss: {test_loss / len(test_loader):.4f}, '
          f'Test Acc: {test_accuracy:.2f}%')

train()

# %%

# Close wandb run
wandb.finish()
