from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch


app = Flask(__name__)

CORS(app)
# Load model
model_path = "./bert-quiz-model2"
 # folder where you saved your model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
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
