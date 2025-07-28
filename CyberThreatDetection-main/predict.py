import os
import re
import pickle
import tensorflow as tf
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np

# Ensure stopwords are available
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load trained model & vectorizer
model_path = "model/attack_model.h5"
vectorizer_path = "model/vectorizer.pkl"
attack_csv_path = "model/attack_mapping.csv"  # ATT&CK mapping file

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path) or not os.path.exists(attack_csv_path):
    raise FileNotFoundError("Model, vectorizer, or ATT&CK mapping file is missing. Train the model first.")

model = tf.keras.models.load_model(model_path)
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Load ATT&CK mapping file
attack_data = pd.read_csv(attack_csv_path)


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text


# Function to predict the TTP technique
def predict_ttp(text):
    processed_text = preprocess_text(text)  # Preprocess input text
    input_vector = vectorizer.transform([processed_text]).toarray()  # Convert to TF-IDF vector
    prediction = model.predict(input_vector)  # Get model predictions
    predicted_ttp_id = np.argmax(prediction, axis=1)[0]  # Extract top prediction

    # Lookup ATT&CK mapping
    attack_entry = attack_data.iloc[predicted_ttp_id]
    technique_id = attack_entry["technique_id"]
    technique_name = attack_entry["name"]
    tactic = attack_entry["tactic"]
    severity = "High" if "persistence" in text else "Medium"  # Simple logic for severity

    # Structured Output
    report = {
        "Threat Description": text,
        "ATT&CK Tactic": tactic,
        "ATT&CK Technique": f"{technique_name} ({technique_id})",
        "Severity": severity
    }
    return report


# Example usage
example_text = "Attackers successfully gained access to domain administrator credentials and established persistence."

prediction = predict_ttp(example_text)
df = pd.DataFrame([prediction])  # Convert to Pandas DataFrame for structured output

# Display as Table
print(df)

# Save as CSV
df.to_csv("threat_report.csv", index=False)

# Save as JSON
df.to_json("threat_report.json", orient="records")

print("\nThreat report generated successfully!")
