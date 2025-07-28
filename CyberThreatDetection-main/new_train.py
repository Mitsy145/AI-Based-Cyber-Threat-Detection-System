import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Check GPU availability
device = torch.device("cuda")
print(f"🚀 Using device: {device}")

# Load dataset
df = pd.read_csv("train.csv")  # Change filename if needed

# Encode labels (Map tactic names to numbers)
label_mapping = {label: idx for idx, label in enumerate(df["tactic"].unique())}
df["tactic"] = df["tactic"].map(label_mapping)

# Save label mapping for later use
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["description"], df["tactic"], test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert labels to tensors
train_labels = torch.tensor(y_train.tolist(), dtype=torch.long).to(device)
test_labels = torch.tensor(y_test.tolist(), dtype=torch.long).to(device)

# Define dataset class
class CyberThreatDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: val.to(device) for key, val in encodings.items()}  # Move to GPU
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Create datasets
train_dataset = CyberThreatDataset(train_encodings, train_labels)
test_dataset = CyberThreatDataset(test_encodings, test_labels)

# Load model to GPU
num_labels = len(label_mapping)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # Disable WandB logging
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("🚀 Training model on GPU...")
trainer.train()

# Save model and tokenizer
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")
print("✅ Model saved successfully in 'model/' folder!")
