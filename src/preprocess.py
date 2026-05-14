"""
WELFake veri setini okur, temizler ve Train/Val/Test olarak böler.

Kullanım:
    python -m src.preprocess
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV,
    COL_TITLE, COL_TEXT, COL_LABEL,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Ham WELFake CSV'yi yükler."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  HATA: Veri seti bulunamadı!\n"
            f"  Beklenen konum: {csv_path}\n\n"
            f"  Çözüm:\n"
            f"  1. https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            f"     adresinden 'WELFake_Dataset.csv' dosyasını indir.\n"
            f"  2) Dosyayı şu konuma taşı:\n"
            f"     {csv_path}\n"
            f"{'='*60}"
        )
    df = pd.read_csv(csv_path)
    logger.info(f"Ham veri yüklendi: {len(df):,} satır, sütunlar: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temel temizleme adımları:
      - Gereksiz sütunları düşür
      - Boş başlık / metin satırlarını kaldır
      - Etiket kontrolü (sadece 0 ve 1)
      - title + text birleştirerek 'content' sütunu oluştur
    """
    # İndeks sütununu kaldır (varsa)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Gerekli sütunlar
    required = [COL_TITLE, COL_TEXT, COL_LABEL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}. Mevcut: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[COL_TITLE, COL_TEXT, COL_LABEL])
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    df = df[df[COL_LABEL].isin([0, 1])]
    after = len(df)
    logger.info(f"Temizleme: {before - after:,} satır kaldırıldı → {after:,} satır kaldı.")

    # Başlık + metin birleştir → modele tek girdi olarak verilecek
    df["content"] = df[COL_TITLE].str.strip() + " [SEP] " + df[COL_TEXT].str.strip()

    logger.info(f"Etiket dağılımı:\n{df[COL_LABEL].value_counts().to_string()}")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    """Train / Validation / Test'e böl ve kaydet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 80 % train — 20 % temp
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_RATIO, stratify=df[COL_LABEL], random_state=RANDOM_SEED
    )
    # temp'i yarı yarıya böl → 10 % val, 10 % test
    val_ratio_of_temp = VAL_RATIO / (1 - TRAIN_RATIO)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio_of_temp, stratify=temp_df[COL_LABEL], random_state=RANDOM_SEED
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,   index=False)
    test_df.to_csv(TEST_CSV,  index=False)

    logger.info(
        f"Bölme tamamlandı:\n"
        f"  Train : {len(train_df):,} satır → {TRAIN_CSV}\n"
        f"  Val   : {len(val_df):,}  satır → {VAL_CSV}\n"
        f"  Test  : {len(test_df):,}  satır → {TEST_CSV}"
    )


def main():
    df = load_raw(RAW_CSV)
    df = clean(df)
    split_and_save(df)
    logger.info("✅ Ön işleme tamamlandı.")


if __name__ == "__main__":
    main()
