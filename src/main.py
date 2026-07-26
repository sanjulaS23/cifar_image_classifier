import torch
from dataset import train_transform, get_dataloaders
from model import CIFAR10_CNN
from train import train_model

def main():
# 1. Checks if GPU is present (if not, CPU is used)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

# 2. Loading the data
    print("Downloading and processing data (CIFAR-10)...")
    train_loader, val_loader, test_loader = get_dataloaders(train_transform, batch_size=128)

# 3. Setting up our CNN model
    model = CIFAR10_CNN(num_classes=10)
    
# 4. Starting to train the model (let's train for 20 epochs)
    print("Training the model will begin...")
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)

if __name__ == '__main__':
    main()