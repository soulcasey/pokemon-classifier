from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch

def get_transform():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform

def get_model(class_names):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    device = get_device()
    model = model.to(device)

    return model

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
