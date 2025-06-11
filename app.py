from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import requests

app = Flask(__name__)
CORS(app)

# 📦 Étape 1 – Télécharger le modèle depuis Hugging Face si absent
model_dir = ".bert-quiz-model-fin/bert-quiz-model2"
model_file = f"{model_dir}/model.safetensors"

if not os.path.exists(model_file):
    os.makedirs(model_dir, exist_ok=True)
    print("🔽 Téléchargement du modèle depuis Hugging Face...")
    url = "https://huggingface.co/dohabkh/bert-eval-asd/resolve/main/model.safetensors"
    r = requests.get(url)
    with open(model_file, "wb") as f:
        f.write(r.content)
    print("✅ Modèle téléchargé avec succès.")

# 📦 Étape 2 – Charger le modèle localement
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_with_confidence(question, student_answer):
    text = question + " [SEP] " + student_answer
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = probs[0][1].item()
    return f"✅ Correct ({confidence:.2f})" if confidence > 0.79 else f"❌ Incorrect ({confidence:.2f})"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    question = data.get("question", "")
    student_answer = data.get("student_answer", "")
    prediction = predict_with_confidence(question, student_answer)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)
