"""
BERT model for medical report classification
"""

import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)

        return self.fc(output)
