"""
Modern Transformer Modeli Eğitimi: RoBERTa

Kullanım:
    python3 -m src.train_roberta
"""
import logging
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from src.config import (
    TRAIN_CSV, VAL_CSV,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN,
    MODELS_DIR
)
from src.dataset import WELFakeDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Eğer CUDA (GPU) aktifse ekran kartını kullan, yoksa CPU ile devam et
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RoBERTa için Özel Dizinler
ROBERTA_MODEL_NAME = "roberta-base"
ROBERTA_SAVE_DIR = MODELS_DIR / "roberta_saved_model"
ROBERTA_TOKENIZER_DIR = MODELS_DIR / "roberta_tokenizer"

class RobertaFakeNewsClassifier(nn.Module):
    def __init__(self, pretrained: str = ROBERTA_MODEL_NAME, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained)
        hidden_size = self.roberta.config.hidden_size # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa'da da [CLS] token'ı 0. indistir (fakat özel token adı <s> dir)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def train_epoch(model, loader, optimizer, scheduler, criterion, epoch):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), lbls)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader)
    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_epoch(model, loader, criterion, epoch):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in pbar:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, lbls)

        total_loss += loss.item()
        total_acc  += accuracy(logits, lbls)

    n = len(loader)
    return total_loss / n, total_acc / n

def main():
    logger.info(f"=== RoBERTa Eğitim Döngüsü Başlıyor ===")
    logger.info(f"Cihaz: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    ROBERTA_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(ROBERTA_TOKENIZER_DIR))
    
    logger.info("RoBERTa Tokenizer kaydedildi ve veri setleri RAM'e yükleniyor...")
    train_ds = WELFakeDataset(TRAIN_CSV, tokenizer, MAX_SEQ_LEN)
    val_ds   = WELFakeDataset(VAL_CSV,   tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info("Yapay zeka modeli (RoBERTa-base) yükleniyor (Yaklaşık 500 MB)...")
    model = RobertaFakeNewsClassifier().to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)  # 0.01 → 0.05: daha güçlü L2 regularization
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    ROBERTA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Early Stopping parametreleri
    patience       = 2
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, epoch)

        logger.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss:   {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            ckpt_path = ROBERTA_SAVE_DIR / "roberta_best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  ✅ En iyi RoBERTa modeli kaydedildi → {ckpt_path}")
        else:
            patience_count += 1
            logger.info(f"  ⚠️  Val loss iyileşmedi ({patience_count}/{patience})")
            if patience_count >= patience:
                logger.info("  🛑 Early stopping devreye girdi — eğitim erken sonlandırılıyor.")
                break

    logger.info("🎉 RoBERTa eğitimi tamamlandı.")

if __name__ == "__main__":
    main()
