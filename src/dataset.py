import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Average values ​​(Stats) of CIFAR-10 data
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)


# Image transformation methods during training (Augmentation Pipeline)
train_transform = transforms.Compose([
    # 1. Adding 4px padding around the image and cropping back to 32x32
    transforms.RandomCrop(32, padding=4),

    # 2. Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # 3. Rotation up to 15 degrees
    transforms.RandomRotation(degrees=15),

    # 4. Adjusting color brightness and contrast
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),

    # 5. Converting to Tensor
    transforms.ToTensor(),

    # 6. Normalization
    transforms.Normalize(MEAN, STD),

    # 7. Random Erasing (deleting a small section) - helps the model learn more robust features
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

# Convert images to Tensors
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def get_dataloaders(train_transform, batch_size=128):
    # Downloading train data
    full_train = datasets.CIFAR10(root='data/', train=True, download=True, transform=train_transform)

    # Downloading test data
    test_set = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)

    # Out of 50k train images, 5k will be taken for validation
    n_val = 5000
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Divide the data into batches
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader