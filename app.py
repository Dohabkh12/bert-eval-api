from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import requests

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model"
MODEL_FILE = f"{MODEL_DIR}/model.safetensors"

# Télécharger le modèle si pas présent
if not os.path.exists(MODEL_FILE):
    os.makedirs(MODEL_DIR, exist_ok=True)
    url = "https://huggingface.co/dohabkh/bert-eval-asd/resolve/main/model.safetensors"
    r = requests.get(url)
    with open(MODEL_FILE, "wb") as f:
        f.write(r.content)

# Chargement du modèle
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("question", "") + " [SEP] " + data.get("student_answer", "")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = probs[0][1].item()
    result = "✅ Correct" if confidence > 0.79 else "❌ Incorrect"
    return jsonify({"result": result, "confidence": round(confidence, 2)})

@app.route("/", methods=["GET"])
def home():
    return "✅ BERT API is running on HF Spaces!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
