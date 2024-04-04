import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader


# Define transforms for FashionMNIST dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),  # Resize images to match ResNet101 input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

subset_train = Subset(train_dataset, range(10000))
subset_test = Subset(test_dataset, range(5000))

# Define data loaders
train_loader = DataLoader(subset_train, batch_size=200, shuffle=True,pin_memory=True)
test_loader = DataLoader(subset_test, batch_size=200, shuffle=False,pin_memory=True)

# Load pre-trained ResNet101 model
resnet = models.resnet101(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in resnet.parameters():
    param.requires_grad = True

# Modify the last fully connected layer to match FashionMNIST (10 classes)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)  # 10 classes in FashionMNIST

# Define optimizers
optimizers = {
    'Adam': optim.Adam(resnet.parameters(), lr=0.001),
    'Adagrad': optim.Adagrad(resnet.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(resnet.parameters(), lr=0.001)
}

# Define loss function
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resnet.to(device)

# Train the model with different optimizers
results = {}
for optimizer_name, optimizer in optimizers.items():
    print(f"Training with {optimizer_name} optimizer...")
    resnet.train()
    resnet.to(device)
    if optimizer_name == 'Adam':
        optimizer_cls = optim.Adam
    elif optimizer_name == 'Adagrad':
        optimizer_cls = optim.Adagrad
    elif optimizer_name == 'RMSprop':
        optimizer_cls = optim.RMSprop
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer = optimizer_cls(resnet.parameters(), lr=0.001)  # Reinitialize optimizer
    losses = []
    accuracies = []
    for epoch in range(3):  # 3 epochs for demonstration
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/3], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    results[optimizer_name] = {
        'loss': losses,
        'accuracy': accuracies
    }

    # Plot curves for training loss and training accuracy
plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(result['loss'], label=f'{optimizer_name} Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for optimizer_name, result in results.items():
    plt.plot(result['accuracy'], label=f'{optimizer_name} Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test the model and report top-5 test accuracy
resnet.eval()
top5_correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.topk(outputs, k=5, dim=1)  # Get top-5 predictions
        # Check if any of the top-5 predictions match the true label
        for i in range(len(labels)):
            if labels[i] in predicted[i]:
                top5_correct += 1
        total += labels.size(0)

top5_accuracy = top5_correct / total
print(f'Top-5 Test Accuracy: {top5_accuracy:.4f}')


