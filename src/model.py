"""
DistilBERT tabanlı ikili sınıflandırma modeli.
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel

from src.config import PRETRAINED_MODEL, DROPOUT_RATE


class FakeNewsClassifier(nn.Module):
    """
    DistilBERT [CLS] token çıktısı üzerine bir classification head ekler.
    Çıktı: 2 sınıf logit (0=Sahte, 1=Gerçek)
    """

    def __init__(
        self,
        pretrained: str = PRETRAINED_MODEL,
        num_classes: int = 2,
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.bert    = DistilBertModel.from_pretrained(pretrained)
        hidden_size  = self.bert.config.hidden_size   # 768 for distilbert-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [CLS]
        logits    = self.classifier(cls_token)
        return logits
