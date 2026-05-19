import logging
import os
import pickle
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.config import (
    TRAIN_CSV, TEST_CSV, MODELS_DIR, FIGURES_DIR,
    COL_TITLE, COL_TEXT, COL_LABEL, LABEL_MAP
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LR_MODEL_DIR = MODELS_DIR / "logistic_regression"
LR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data(csv_path: Path):
    logger.info(f"Veri yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    
    df[COL_TITLE] = df[COL_TITLE].fillna("")
    df[COL_TEXT] = df[COL_TEXT].fillna("")
    
    X = df[COL_TITLE] + " " + df[COL_TEXT]
    y = df[COL_LABEL]
    
    return X, y


def main():
    logger.info("=== Makine Öğrenmesi (Logistic Regression) Eğitimi Başlıyor ===")
    
    X_train, y_train = load_and_prepare_data(TRAIN_CSV)
    X_test, y_test = load_and_prepare_data(TEST_CSV)
    
    logger.info("Metinler TF-IDF ile sayısallaştırılıyor (max_features=15000)...")
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info("Logistic Regression modeli eğitiliyor...")
    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    model.fit(X_train_vec, y_train)
    
    logger.info("Test seti üzerinde değerlendirme yapılıyor...")
    preds = model.predict(X_test_vec)
    
    target_names = [LABEL_MAP[k] for k in sorted(LABEL_MAP)]
    report = classification_report(y_test, preds, target_names=target_names)
    logger.info(f"\nSınıflandırma Raporu (Logistic Regression):\n{report}")
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title("Confusion Matrix — Logistic Regression")
    
    fig_path = FIGURES_DIR / "confusion_matrix_lr.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Hata matrisi grafiği kaydedildi → {fig_path}")
    
    model_path = LR_MODEL_DIR / "lr_model.pkl"
    vec_path = LR_MODEL_DIR / "tfidf_vectorizer.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    logger.info(f"✅ Eğitim tamamlandı. Model ve Vektörizer kaydedildi → {LR_MODEL_DIR}")


if __name__ == "__main__":
    main()
