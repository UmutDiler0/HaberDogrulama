"""
Proje genelinde kullanılan sabit değerler ve yol yapılandırmaları.
"""
from pathlib import Path

# ─── Proje Kök Dizini ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Veri Dizinleri ──────────────────────────────────────────────────────────
DATA_DIR         = ROOT_DIR / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"

# WELFake ham CSV dosyası (Kaggle'dan indirip buraya koy)
RAW_CSV          = RAW_DATA_DIR / "WELFake_Dataset.csv"

# Train / Validation / Test CSV dosyaları
TRAIN_CSV        = PROCESSED_DIR / "train.csv"
VAL_CSV          = PROCESSED_DIR / "val.csv"
TEST_CSV         = PROCESSED_DIR / "test.csv"

# ─── Model Dizinleri ─────────────────────────────────────────────────────────
MODELS_DIR       = ROOT_DIR / "models"
TOKENIZER_DIR    = MODELS_DIR / "tokenizer"
MODEL_SAVE_DIR   = MODELS_DIR / "saved_model"
TFLITE_PATH      = MODELS_DIR / "fake_news_detector.tflite"

# ─── Çıktı Dizini ────────────────────────────────────────────────────────────
OUTPUTS_DIR      = ROOT_DIR / "outputs"
FIGURES_DIR      = OUTPUTS_DIR / "figures"

# ─── Veri Seti Parametreleri ─────────────────────────────────────────────────
# WELFake sütun adları
COL_TITLE  = "title"
COL_TEXT   = "text"
COL_LABEL  = "label"   # 0 = Sahte (Fake), 1 = Gerçek (Real)

# Train / Val / Test oranları
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ─── Model Hiper-Parametreleri ────────────────────────────────────────────────
MAX_SEQ_LEN   = 256       # Token sayısı üst sınırı
BATCH_SIZE    = 32
EPOCHS        = 5
LEARNING_RATE = 2e-5
DROPOUT_RATE  = 0.3

# Kullanılacak ön-eğitimli model (HuggingFace Hub)
PRETRAINED_MODEL = "distilbert-base-uncased"

# WELFake orijinal veri setinde 1=Fake (Sahte), 0=Real (Gerçek) kabul edilir
LABEL_MAP = {0: "GERÇEK", 1: "SAHTE"}
