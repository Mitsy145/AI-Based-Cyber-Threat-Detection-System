import os
import re
import pickle
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import numpy as np

# Ensure stopwords are available
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load trained model & vectorizer
model_path = "model/attack_model.h5"
vectorizer_path = "model/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file is missing. Train the model first.")

model = tf.keras.models.load_model(model_path)
vectorizer = pickle.load(open(vectorizer_path, "rb"))

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
    predicted_ttp = np.argmax(prediction, axis=1)[0]  # Extract top prediction
    return f"TTP-{predicted_ttp}"  # Format as TTP-ID

# Example usage
example_text = "Bro i think my wifi is hacked with evil twin"

prediction = predict_ttp(example_text)
print(f"Predicted TTP: {prediction}")