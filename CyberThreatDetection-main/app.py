from flask import Flask, render_template, request, session, redirect, url_for
import os
import re
import pickle
import tensorflow as tf
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

model_path = "model/attack_model.h5"
vectorizer_path = "model/vectorizer.pkl"
attack_csv_path = "model/attack_mapping.csv"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path) or not os.path.exists(attack_csv_path):
    raise FileNotFoundError("Model, vectorizer, or ATT&CK mapping file is missing. Train the model first.")

model = tf.keras.models.load_model(model_path)
vectorizer = pickle.load(open(vectorizer_path, "rb"))

attack_data = pd.read_csv(attack_csv_path)

app = Flask(__name__)
app.secret_key = "supersecretkey"

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def predict_ttp(text):
    processed_text = preprocess_text(text)
    input_vector = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(input_vector)  # Get model predictions
    predicted_ttp_id = np.argmax(prediction, axis=1)[0]

    # Lookup ATT&CK mapping
    attack_entry = attack_data.iloc[predicted_ttp_id]
    technique_id = attack_entry["technique_id"]
    technique_name = attack_entry["name"]
    tactic = attack_entry["tactic"]
    if any(keyword in text.lower() for keyword in ["persistence", "privilege escalation", "ransomware", "critical"]):
        severity = "High"
    elif any(keyword in text.lower() for keyword in ["brute-force", "phishing", "malware", "suspicious"]):
        severity = "Medium"
    else:
        severity = "Low"

    # Structured Output
    report = {
        "Threat Description": text,
        "ATT&CK Tactic": tactic,
        "ATT&CK Technique": f"{technique_name} ({technique_id})",
        "Severity": severity
    }
    return report


@app.route("/", methods=["GET", "POST"])
def index():
    if "results" not in session:
        session["results"] = []

    if request.method == "POST":
        text = request.form["threat_text"]
        result = predict_ttp(text)
        session["results"].append(result)
        session.modified = True

    return render_template("index.html", results=session["results"])


@app.route("/clear", methods=["POST"])
def clear_results():
    session["results"] = []
    session.modified = True
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
