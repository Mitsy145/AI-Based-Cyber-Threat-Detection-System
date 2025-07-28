import torch
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset and structure properly
def load_data():
    data_files = {"train": "train.csv", "test": "test.csv", "validation": "val.csv"}

    datasets = {}
    for split, file in data_files.items():
        df = pd.read_csv(file)

        # Ensure the necessary columns exist
        if "name" not in df.columns or "description" not in df.columns or "tactic" not in df.columns:
            raise ValueError("CSV file must contain 'name', 'description', and 'tactic' columns.")

        # Combine 'name' and 'description' into a single text column
        df["text"] = df["name"] + " - " + df["description"]

        # Create a unique mapping for tactics to labels
        unique_tactics = df["tactic"].unique()
        label_map = {tactic: idx for idx, tactic in enumerate(unique_tactics)}

        # Apply label mapping
        df["labels"] = df["tactic"].map(label_map)

        # Convert to Hugging Face Dataset
        datasets[split] = Dataset.from_pandas(df[["text", "labels"]])

    return DatasetDict(datasets), label_map


# Preprocess data
def preprocess_data(dataset, tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        tokenized_inputs["labels"] = examples["labels"]  # Direct label mapping
        return tokenized_inputs

    return dataset.map(tokenize_and_align_labels, batched=True)


# Initialize tokenizer and model
def initialize_model(num_labels):
    model_name = "bert-base-uncased"  # Use a cybersecurity-specific BERT if available
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


# Define the training loop
def train_model(tokenizer, model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()


# Main workflow
if __name__ == "__main__":
    print("Loading dataset...")
    dataset, label_map = load_data()

    print("Initializing model and tokenizer...")
    tokenizer, model = initialize_model(num_labels=len(label_map))

    print("Preprocessing data...")
    processed_dataset = preprocess_data(dataset, tokenizer)

    print("Training model...")
    train_model(
        tokenizer,
        model,
        processed_dataset["train"],
        processed_dataset["validation"],
    )

    print("Saving model...")
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

    print("Training complete.")
