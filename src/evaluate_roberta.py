"""
Test seti üzerinde RoBERTa model değerlendirmesi.
Accuracy, Precision, Recall, F1 ve Confusion Matrix raporlanır.

Kullanım:
    python -m src.evaluate_roberta
"""
import logging
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    TEST_CSV, BATCH_SIZE, MAX_SEQ_LEN, LABEL_MAP, FIGURES_DIR,
)
from src.dataset import WELFakeDataset
from src.train_roberta import RobertaFakeNewsClassifier, ROBERTA_SAVE_DIR, ROBERTA_TOKENIZER_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")

def main():
    logger.info("=== RoBERTa Test Süreci Başlıyor ===")
    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_TOKENIZER_DIR))
    test_ds   = WELFakeDataset(TEST_CSV, tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model yükle
    model = RobertaFakeNewsClassifier().to(DEVICE)
    ckpt  = ROBERTA_SAVE_DIR / "roberta_best_model.pt"
    
    # Eğitilmiş ağırlıkları yükle
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
    logger.info(f"\nSınıflandırma Raporu (RoBERTa):\n{report}")

    # Confusion Matrix görseli
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm  = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Purples") # RoBERTa için mor renk tonu
    ax.set_title("Confusion Matrix — RoBERTa (Test Seti)")
    
    fig_path = FIGURES_DIR / "confusion_matrix_roberta.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix kaydedildi → {fig_path}")

if __name__ == "__main__":
    main()
