import logging
from pathlib import Path
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup

from src.config import (
    TRAIN_CSV, VAL_CSV,
    PRETRAINED_MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_SEQ_LEN,
    MODEL_SAVE_DIR, TOKENIZER_DIR,
)
from src.dataset import WELFakeDataset
from src.model import FakeNewsClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    logger.info(f"Cihaz: {DEVICE}")

    tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    logger.info(f"Tokenizer kaydedildi → {TOKENIZER_DIR}")

    logger.info("Veri setleri RAM'e yükleniyor (Bu işlem birkaç saniye sürebilir)...")
    train_ds = WELFakeDataset(TRAIN_CSV, tokenizer, MAX_SEQ_LEN)
    val_ds   = WELFakeDataset(VAL_CSV,   tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds):,} örnek | Val: {len(val_ds):,} örnek")

    logger.info("Yapay zeka modeli (DistilBERT) yükleniyor / indiriliyor (Yaklaşık 260 MB)...")
    model = FakeNewsClassifier().to(DEVICE)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i, block in enumerate(model.bert.transformer.layer):
        if i < 4:
            for param in block.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Eğitilecek parametre: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.05
    )
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
            ckpt_path = MODEL_SAVE_DIR / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  ✅ En iyi model kaydedildi → {ckpt_path}")
        else:
            patience_count += 1
            logger.info(f"  ⚠️  Val loss iyileşmedi ({patience_count}/{patience})")
            if patience_count >= patience:
                logger.info("  🛑 Early stopping devreye girdi — eğitim erken sonlandırılıyor.")
                break

    logger.info("🎉 Eğitim tamamlandı.")


if __name__ == "__main__":
    main()
