"""
ResNet18 Transfer Learning Model
Supports:
- Frozen feature extraction
- Full fine-tuning
- Custom classification head
"""

import torch.nn as nn
import torchvision.models as models


class ResNetTransfer(nn.Module):
    def __init__(self, mode="frozen"):
        """
        mode:
            "frozen"   → Feature extraction
            "finetune" → Full fine-tuning
        """
        super(ResNetTransfer, self).__init__()

        # Load pretrained ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace final classification layer
        num_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

        # Freeze or unfreeze layers
        if mode == "frozen":
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Unfreeze classifier head
            for param in self.base_model.fc.parameters():
                param.requires_grad = True

        elif mode == "finetune":
            for param in self.base_model.parameters():
                param.requires_grad = True

        else:
            raise ValueError("Mode must be 'frozen' or 'finetune'")

    def forward(self, x):
        return self.base_model(x)
