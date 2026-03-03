"""
CNN Model using Transfer Learning (ResNet18)
CPU friendly version
"""

import torch.nn as nn
import torchvision.models as models


class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXrayCNN, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Freeze early layers (important for CPU training)
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
