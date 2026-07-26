import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A collection of layers that read the shapes of images"""
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     # Reduces the size of the image by half
            nn.Dropout2d(dropout)
        )
    def forward(self, x):
        return self.block(x)

class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Analyzes the image through 3 layers (Blocks)
        self.features = nn.Sequential(
            ConvBlock(3,   64,  dropout=0.2),
            ConvBlock(64,  128, dropout=0.3),
            ConvBlock(128, 256, dropout=0.4)
        )
# The final decision-making part (Classifier)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      
            nn.Flatten(),                  
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Returns the result for 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x