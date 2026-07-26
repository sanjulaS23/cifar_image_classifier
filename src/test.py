import torch
import torchvision
import torchvision.transforms as transforms
from model import CIFAR10_CNN  # If running locally, or use your model class definition
import matplotlib.pyplot as plt
import random

# 1. Classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Setup device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10_CNN(num_classes=10).to(device)

checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Model loaded successfully with Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")

# 3. Load Test Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 4. Pick a random image and predict
index = random.randint(0, len(testset) - 1)
image, label = testset[index]

# Add batch dimension and move to device
input_image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_image)
    _, predicted = torch.max(outputs, 1)

print(f"True Label: {classes[label]}")
print(f"Predicted Label: {classes[predicted.item()]}")