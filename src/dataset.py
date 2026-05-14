"""
PyTorch Dataset sınıfı — DistilBERT tokenizer ile WELFake verilerini hazırlar.
"""
import csv
import sys
import torch
from torch.utils.data import Dataset

from src.config import COL_LABEL, MAX_SEQ_LEN

# Haber metinleri çok uzun olabileceği için CSV limitini artırıyoruz (Windows C long limitine uygun)
csv.field_size_limit(2147483647)

class WELFakeDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str | Path
        train.csv / val.csv / test.csv yolu
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer nesnesi
    max_len : int
        Token üst sınırı (varsayılan config.MAX_SEQ_LEN)
    """

    def __init__(self, csv_path, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.texts = []
        self.labels = []
        
        # Pandas yerine standart CSV okuyucu ile sadece gerekli verileri alıyoruz
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(str(row["content"]))
                self.labels.append(int(row[COL_LABEL]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }
