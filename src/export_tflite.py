"""
Eğitilmiş modeli TensorFlow Lite formatına dönüştürür.
Android uygulamasında kullanılmak üzere .tflite dosyası üretir.

Kullanım:
    python -m src.export_tflite

NOT: Bu script pytorch → onnx → tensorflow → tflite zincirini kullanır.
Gerekli paketler: onnx, onnx-tf, tensorflow (requirements-export.txt)
"""
import logging
from pathlib import Path

import torch
from transformers import DistilBertTokenizerFast

from src.config import (
    MODEL_SAVE_DIR, TOKENIZER_DIR, TFLITE_PATH, MAX_SEQ_LEN, PRETRAINED_MODEL,
)
from src.model import FakeNewsClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")  # Export CPU'da yapılır
ONNX_PATH = MODEL_SAVE_DIR / "fake_news_detector.onnx"


def export_onnx(model: FakeNewsClassifier, onnx_path: Path) -> None:
    """PyTorch modelini ONNX formatına dönüştür."""
    model.eval()
    dummy_ids  = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits":         {0: "batch_size"},
        },
        opset_version=13,
    )
    logger.info(f"ONNX modeli kaydedildi → {onnx_path}")


def onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> None:
    """ONNX → TensorFlow → TFLite dönüşüm zinciri."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as e:
        logger.error(
            f"Eksik paket: {e}\n"
            "Lütfen şunu çalıştır: pip install onnx onnx-tf tensorflow"
        )
        return

    tf_model_path = MODEL_SAVE_DIR / "tf_model"
    onnx_model = onnx.load(str(onnx_path))
    tf_rep     = prepare(onnx_model)
    tf_rep.export_graph(str(tf_model_path))
    logger.info(f"TF SavedModel → {tf_model_path}")

    # TFLite dönüşümü
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]          # INT8 kuantizasyon
    tflite_model = converter.convert()

    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    logger.info(f"✅ TFLite modeli kaydedildi → {tflite_path}")


def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))  # noqa: F841
    model = FakeNewsClassifier().to(DEVICE)
    ckpt  = MODEL_SAVE_DIR / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    export_onnx(model, ONNX_PATH)
    onnx_to_tflite(ONNX_PATH, TFLITE_PATH)


if __name__ == "__main__":
    main()
