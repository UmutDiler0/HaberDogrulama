#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  HaberDogrulama — Ubuntu Kurulum & Eğitim Scripti
#  Kullanım: bash setup_ubuntu.sh
# ═══════════════════════════════════════════════════════════════════════

set -e  # Herhangi bir hata olursa scripti durdur

echo "╔══════════════════════════════════════════════╗"
echo "║   HaberDogrulama Ubuntu Kurulum Scripti     ║"
echo "╚══════════════════════════════════════════════╝"

# ─── 1. Sistem Paketleri ─────────────────────────────────────────────
echo ""
echo "📦 [1/5] Sistem paketleri güncelleniyor..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv git

# ─── 2. Sanal Ortam Kur ──────────────────────────────────────────────
echo ""
echo "🐍 [2/5] Python sanal ortamı (venv) oluşturuluyor..."
python3 -m venv venv
source venv/bin/activate
echo "   ✅ Sanal ortam aktif: $(which python)"

# ─── 3. Python Bağımlılıkları ────────────────────────────────────────
echo ""
echo "📥 [3/5] Python paketleri kuruluyor (Bu işlem birkaç dakika sürebilir)..."

# Pip'i güncelle
pip install --upgrade pip -q

# PyTorch — GPU varsa CUDA sürümünü kur, yoksa CPU sürümünü kur
if command -v nvidia-smi &> /dev/null; then
    echo "   🎮 NVIDIA GPU tespit edildi! CUDA destekli PyTorch kuruluyor..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    echo "   💻 GPU bulunamadı. CPU sürümü PyTorch kuruluyor..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# Diğer bağımlılıklar
pip install -r requirements.txt -q
pip install flask -q  # Web arayüzü için

echo "   ✅ Tüm paketler kuruldu."

# ─── 4. Eski Model Dosyalarını Yedekle (Silme!) ──────────────────────
echo ""
echo "💾 [4/5] Eski model dosyaları yedekleniyor..."

BACKUP_DIR="models/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "models/saved_model/best_model.pt" ]; then
    cp "models/saved_model/best_model.pt" "$BACKUP_DIR/"
    echo "   ✅ DistilBERT modeli yedeklendi → $BACKUP_DIR/best_model.pt"
else
    echo "   ℹ️  DistilBERT model dosyası bulunamadı, atlanıyor."
fi

if [ -f "models/roberta_saved_model/roberta_best_model.pt" ]; then
    cp "models/roberta_saved_model/roberta_best_model.pt" "$BACKUP_DIR/"
    echo "   ✅ RoBERTa modeli yedeklendi → $BACKUP_DIR/roberta_best_model.pt"
else
    echo "   ℹ️  RoBERTa model dosyası bulunamadı, atlanıyor."
fi

# ─── 5. Eğitim Seçenekleri ───────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Hangi modeli eğitmek istersiniz?                      ║"
echo "║                                                          ║"
echo "║   1) Sadece DistilBERT                                  ║"
echo "║   2) Sadece RoBERTa                                     ║"
echo "║   3) Önce ML modeller (LR + DT), sonra DistilBERT      ║"
echo "║   4) Tüm modelleri sırayla eğit (Tam Pipeline)          ║"
echo "║   5) Sadece web uygulamasını başlat (eğitim yapma)      ║"
echo "╚══════════════════════════════════════════════════════════╝"
read -p "Seçiminiz [1-5]: " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "🚀 DistilBERT eğitimi başlıyor..."
        python -m src.train
        echo "📊 DistilBERT değerlendirmesi yapılıyor..."
        python -m src.evaluate
        ;;
    2)
        echo ""
        echo "🚀 RoBERTa eğitimi başlıyor..."
        python -m src.train_roberta
        echo "📊 RoBERTa değerlendirmesi yapılıyor..."
        python -m src.evaluate_roberta
        ;;
    3)
        echo ""
        echo "🚀 ML modelleri (LR + DT) eğitimi başlıyor..."
        python -m src.train_ml
        echo "🚀 DistilBERT eğitimi başlıyor..."
        python -m src.train
        echo "📊 DistilBERT değerlendirmesi yapılıyor..."
        python -m src.evaluate
        ;;
    4)
        echo ""
        echo "🚀 [1/4] ML modelleri (Logistic Regression + Decision Tree) eğitiliyor..."
        python -m src.train_ml
        echo ""
        echo "🚀 [2/4] DistilBERT eğitimi başlıyor..."
        python -m src.train
        echo "📊 [3/4] DistilBERT değerlendirmesi yapılıyor..."
        python -m src.evaluate
        echo ""
        echo "🚀 [4/4] RoBERTa eğitimi başlıyor..."
        python -m src.train_roberta
        echo "📊 RoBERTa değerlendirmesi yapılıyor..."
        python -m src.evaluate_roberta
        ;;
    5)
        echo ""
        echo "⏭️  Eğitim atlanıyor, doğrudan web uygulaması başlatılıyor..."
        ;;
    *)
        echo "❌ Geçersiz seçim. Çıkılıyor."
        exit 1
        ;;
esac

# ─── Web Uygulamasını Başlat ─────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Web uygulaması başlatılsın mı?            ║"
echo "╚══════════════════════════════════════════════╝"
read -p "Uygulamayı başlat? [e/h]: " START_APP

if [[ "$START_APP" == "e" || "$START_APP" == "E" ]]; then
    echo ""
    echo "🌐 Web uygulaması başlatılıyor → http://localhost:5000"
    python app.py
fi

echo ""
echo "✅ Kurulum tamamlandı!"
