"""
Eğitilmiş modeli tek bir haber metni üzerinde test etmek için kullanılan
basit çıkarım (inference) scripti.

Kullanım:
    python -m src.predict --title "Başlık" --text "Makale gövdesi..."
    python -m src.predict --text "Sadece metin de verilebilir"
"""
import argparse
import torch
from transformers import DistilBertTokenizerFast

from src.config import MODEL_SAVE_DIR, TOKENIZER_DIR, MAX_SEQ_LEN, LABEL_MAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    from src.model import FakeNewsClassifier
    model = FakeNewsClassifier().to(DEVICE)
    ckpt  = MODEL_SAVE_DIR / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def predict(title: str, text: str, model, tokenizer) -> dict:
    content = f"{title.strip()} [SEP] {text.strip()}" if title else text.strip()

    enc = tokenizer(
        content,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(ids, mask)
        probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return {
        "label":      LABEL_MAP[pred_idx],
        "confidence": round(confidence * 100, 2),
        "probs":      {LABEL_MAP[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)},
    }


def main():
    parser = argparse.ArgumentParser(description="Haber Doğrulama Tahmini")
    parser.add_argument("--title", type=str, default="", help="Haberin başlığı (opsiyonel)")
    parser.add_argument("--text",  type=str, required=True, help="Haberin içeriği")
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    model     = load_model()
    result    = predict(args.title, args.text, model, tokenizer)

    print("\n" + "="*50)
    print(f"  Tahmin   : {result['label']}")
    print(f"  Güven    : %{result['confidence']}")
    print(f"  Olasılıklar:")
    for lbl, prob in result["probs"].items():
        print(f"    {lbl:8s}: %{prob}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
