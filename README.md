# HaberDoğrulama — WELFake Sahte Haber Tespit Modeli

WELFake veri seti kullanılarak eğitilmiş ikili sınıflandırma modeli.  
**DistilBERT** tabanlı bir transformer mimarisi kullanır (0 = Sahte, 1 = Gerçek).

---

## 📁 Dosya Yapısı

```
HaberDogrulama/
├── data/
│   ├── raw/
│   │   └── WELFake_Dataset.csv   ← ✅ BURAYA KOY
│   └── processed/                ← Otomatik oluşturulur
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── tokenizer/                ← Eğitim sonrası otomatik oluşur
│   ├── saved_model/
│   │   └── best_model.pt
│   └── fake_news_detector.tflite
│
├── outputs/
│   └── figures/
│       └── confusion_matrix.png
│
├── src/
│   ├── config.py       ← Tüm ayarlar burada
│   ├── preprocess.py   ← Veri temizleme & bölme
│   ├── dataset.py      ← PyTorch Dataset sınıfı
│   ├── model.py        ← DistilBERT model tanımı
│   ├── train.py        ← Eğitim döngüsü
│   ├── evaluate.py     ← Test seti değerlendirmesi
│   ├── predict.py      ← Tek haber tahmini (CLI)
│   └── export_tflite.py ← Android için TFLite export
│
├── requirements.txt
├── requirements-export.txt
└── README.md
```

---

## 🚀 Hızlı Başlangıç

### 1. Veri Setini İndir

[Kaggle — WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) sayfasına git ve `WELFake_Dataset.csv` dosyasını indir.  
İndirilen dosyayı şu konuma taşı:

```
data/raw/WELFake_Dataset.csv
```

### 2. Ortamı Hazırla

```powershell
# Sanal ortamı etkinleştir
.\venv\Scripts\Activate.ps1

# Bağımlılıkları kur
pip install -r requirements.txt
```

### 3. Veriyi İşle

```powershell
python -m src.preprocess
```

Bu komut `data/processed/` altında `train.csv`, `val.csv` ve `test.csv` oluşturur.

### 4. Modeli Eğit

```powershell
python -m src.train
```

GPU varsa otomatik algılanır. En iyi model `models/saved_model/best_model.pt` olarak kaydedilir.

### 5. Değerlendir

```powershell
python -m src.evaluate
```

Confusion matrix görseli `outputs/figures/confusion_matrix.png` olarak kaydedilir.

### 6. Tahmin Yap

```powershell
python -m src.predict --title "Başlık" --text "Haber metni buraya..."
```

### 7. Android için Export (İsteğe Bağlı)

```powershell
pip install -r requirements-export.txt
python -m src.export_tflite
```

---

## ⚙️ Parametreler

Tüm hiper-parametreler `src/config.py` dosyasından değiştirilebilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `PRETRAINED_MODEL` | `distilbert-base-uncased` | HuggingFace model ID |
| `MAX_SEQ_LEN` | `256` | Maksimum token sayısı |
| `BATCH_SIZE` | `32` | Batch büyüklüğü |
| `EPOCHS` | `5` | Epoch sayısı |
| `LEARNING_RATE` | `2e-5` | AdamW öğrenme oranı |
| `TRAIN_RATIO` | `0.80` | Train oranı |

---

## 📊 WELFake Veri Seti Hakkında

| Özellik | Değer |
|---|---|
| Toplam örnek | ~72,134 |
| Sütunlar | `title`, `text`, `label` |
| Etiket | 0 = Sahte, 1 = Gerçek |
| Kaynak | Kaggle / Zenodo |