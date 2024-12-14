import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import common

transform = common.get_transform()

# Load the Pokémon dataset
data_dir = "./pokemon"

train_data = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class_names = train_data.classes

# Define the model
model = common.get_model(class_names)
device = common.get_device()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        batch_classes = [class_names[label] for label in labels.cpu().numpy()]

        print(f"Batch Pokémon: {batch_classes}")
    
    print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "pokemon_classifier.pth")
torch.save(class_names, "class_names.pth")
