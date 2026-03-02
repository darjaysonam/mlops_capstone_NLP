"""
Ensemble model combining image and text predictions
"""

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, image_model, text_model, num_classes):
        super(EnsembleModel, self).__init__()

        self.image_model = image_model
        self.text_model = text_model

        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, image, input_ids=None, attention_mask=None):

        image_out = self.image_model(image)

        if input_ids is not None:
            text_out = self.text_model(input_ids, attention_mask)
            combined = torch.cat((image_out, text_out), dim=1)
            return self.fc(combined)

        return image_out
