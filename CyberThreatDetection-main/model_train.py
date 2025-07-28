import os
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train.csv")

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Selecting input (X) and labels (y)
X = df["description"]  # Use the correct column name
y = df["tactic"]  # Use the correct label column

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X).toarray()

# Ensure the 'model' directory exists
os.makedirs("model", exist_ok=True)

# Save vectorizer
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Convert y labels to numerical values
y_train = y_train.astype("category").cat.codes
y_test = y_test.astype("category").cat.codes
num_classes = len(set(y_train))

# Define neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(5000,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer matches num_classes
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save trained model
model.save("model/attack_model.h5")
