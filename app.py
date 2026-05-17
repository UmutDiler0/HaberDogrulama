import os
import pickle
import sqlite3
from datetime import datetime
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path

def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            model_type TEXT,
            title TEXT,
            label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

from src.config import (
    MAX_SEQ_LEN, LABEL_MAP, MODELS_DIR, FIGURES_DIR, TOKENIZER_DIR
)

# Deep Learning Imports
from src.model import FakeNewsClassifier
from transformers import DistilBertTokenizerFast, AutoTokenizer
from src.train_roberta import RobertaFakeNewsClassifier, ROBERTA_SAVE_DIR, ROBERTA_TOKENIZER_DIR

app = Flask(__name__)
DEVICE = torch.device("cpu")

# --- Model Caches (Lazy Loading) ---
LOADED_MODELS = {
    'dt': None,
    'lr': None,
    'distilbert': None,
    'roberta': None
}
LOADED_VEC = {
    'dt': None,
    'lr': None,
    'distilbert': None,
    'roberta': None
}

def load_dt():
    if LOADED_MODELS['dt'] is None:
        with open(MODELS_DIR / "decision_tree" / "dt_model.pkl", "rb") as f:
            LOADED_MODELS['dt'] = pickle.load(f)
        with open(MODELS_DIR / "decision_tree" / "tfidf_vectorizer_dt.pkl", "rb") as f:
            LOADED_VEC['dt'] = pickle.load(f)
    return LOADED_MODELS['dt'], LOADED_VEC['dt']

def load_lr():
    if LOADED_MODELS['lr'] is None:
        with open(MODELS_DIR / "logistic_regression" / "lr_model.pkl", "rb") as f:
            LOADED_MODELS['lr'] = pickle.load(f)
        with open(MODELS_DIR / "logistic_regression" / "tfidf_vectorizer.pkl", "rb") as f:
            LOADED_VEC['lr'] = pickle.load(f)
    return LOADED_MODELS['lr'], LOADED_VEC['lr']

def load_distilbert():
    if LOADED_MODELS['distilbert'] is None:
        model = FakeNewsClassifier().to(DEVICE)
        ckpt = MODELS_DIR / "saved_model" / "best_model.pt"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        LOADED_MODELS['distilbert'] = model
        
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
        LOADED_VEC['distilbert'] = tokenizer
    return LOADED_MODELS['distilbert'], LOADED_VEC['distilbert']

def load_roberta():
    if LOADED_MODELS['roberta'] is None:
        model = RobertaFakeNewsClassifier().to(DEVICE)
        ckpt = ROBERTA_SAVE_DIR / "roberta_best_model.pt"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        LOADED_MODELS['roberta'] = model
        
        tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_TOKENIZER_DIR))
        LOADED_VEC['roberta'] = tokenizer
    return LOADED_MODELS['roberta'], LOADED_VEC['roberta']


# --- Prediction Logic ---
def predict_ml(title, text, model, vectorizer):
    content = f"{title.strip()} {text.strip()}".strip()
    vec = vectorizer.transform([content])
    preds = model.predict(vec)
    probs = model.predict_proba(vec)[0]
    
    pred_idx = preds[0]
    confidence = probs[pred_idx]
    
    return {
        "label": LABEL_MAP[pred_idx],
        "confidence": round(float(confidence) * 100, 2),
        "probs": {LABEL_MAP[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}
    }

def predict_dl(title, text, model, tokenizer):
    content = f"{title.strip()} [SEP] {text.strip()}" if title else text.strip()
    enc = tokenizer(
        content,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1).squeeze()
        
        # Squeeze sonrası tek boyutlu bir tensor değilse (ör: batch_size > 1 ama burada 1), 
        # yine de elemanlara erişebilmek için:
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    return {
        "label": LABEL_MAP[pred_idx],
        "confidence": round(confidence * 100, 2),
        "probs": {LABEL_MAP[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)},
    }


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    model_type = data.get("model_type", "distilbert")
    title = data.get("title", "")
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Metin (text) alanı zorunludur."}), 400
        
    try:
        if model_type == "dt":
            m, v = load_dt()
            res = predict_ml(title, text, m, v)
        elif model_type == "lr":
            m, v = load_lr()
            res = predict_ml(title, text, m, v)
        elif model_type == "distilbert":
            m, v = load_distilbert()
            res = predict_dl(title, text, m, v)
        elif model_type == "roberta":
            m, v = load_roberta()
            res = predict_dl(title, text, m, v)
        else:
            return jsonify({"error": "Geçersiz model türü"}), 400
            
        # Veritabanına Kaydet
        try:
            conn = sqlite3.connect('history.db')
            c = conn.cursor()
            c.execute(
                "INSERT INTO predictions (timestamp, model_type, title, label, confidence) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_type, title if title else "Başlıksız", res["label"], res["confidence"])
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            print("DB Error:", db_err)

        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        conn = sqlite3.connect('history.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM predictions ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        
        data = []
        for r in rows:
            data.append(dict(r))
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/figures/<path:filename>")
def serve_figures(filename):
    return send_from_directory(FIGURES_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
