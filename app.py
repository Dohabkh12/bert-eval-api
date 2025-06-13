import os
import zipfile
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# 1. Téléchargement automatique du modèle depuis Google Drive
ZIP_ID = "1fNLZ2sG-gkZ9m6GZ7pOau4cw64vrgO4O"  # ← remplace par ton vrai ID
ZIP_PATH = "bert.zip"
MODEL_DIR = "model"

if not os.path.exists(MODEL_DIR):
    print("📥 Téléchargement du modèle...")
    gdown.download(f"https://drive.google.com/uc?id={ZIP_ID}", ZIP_PATH, quiet=False)

    print("📦 Décompression...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# 2. Chargement du modèle
print("🔁 Chargement du modèle...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3. API de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    question = data.get("question", "")
    student_answer = data.get("student_answer", "")
    text = question + " [SEP] " + student_answer

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs[0][1].item()

    result = f"✅ Correct ({confidence:.2f})" if confidence > 0.79 else f"❌ Incorrect ({confidence:.2f})"
    return jsonify(result)

# 4. Lancement
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
