"""
Test seti üzerinde model değerlendirmesi.
Accuracy, Precision, Recall, F1 ve Confusion Matrix raporlanır.

Kullanım:
    python -m src.evaluate
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from src.config import (
    TEST_CSV, MODEL_SAVE_DIR, TOKENIZER_DIR,
    BATCH_SIZE, MAX_SEQ_LEN, LABEL_MAP, FIGURES_DIR,
)
from src.dataset import WELFakeDataset
from src.model import FakeNewsClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Tokenizer & Dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    test_ds   = WELFakeDataset(TEST_CSV, tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model yükle
    model = FakeNewsClassifier().to(DEVICE)
    ckpt  = MODEL_SAVE_DIR / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    logger.info(f"Model yüklendi: {ckpt}")

    # Tahmin
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].numpy()

            logits = model(ids, mask)
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(lbls)

    # Rapor
    target_names = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    logger.info(f"\nSınıflandırma Raporu:\n{report}")

    # Confusion Matrix görseli
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm  = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — WELFake Test Seti")
    fig_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix kaydedildi → {fig_path}")


if __name__ == "__main__":
    main()
