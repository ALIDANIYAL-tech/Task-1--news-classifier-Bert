"""
News Topic Classifier - Training Script
Task: Fine-tune BERT for news classification
Dataset: AG News (4 categories)
"""

import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

print("="*60)
print("NEWS TOPIC CLASSIFIER - BERT FINE-TUNING")
print("="*60)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load AG News dataset
print("\n[1/5] Loading AG News dataset...")
dataset = load_dataset("ag_news")
print(f"✓ Train samples: {len(dataset['train']):,}")
print(f"✓ Test samples: {len(dataset['test']):,}")

# Initialize BERT tokenizer
print("\n[2/5] Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✓ Tokenizer ready")

# Tokenization function
def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Preprocess dataset
print("\n[3/5] Tokenizing and preprocessing...")
tokenized_train = dataset["train"].map(tokenize_data, batched=True)
tokenized_test = dataset["test"].map(tokenize_data, batched=True)

# Use subset for faster training
tokenized_train = tokenized_train.select(range(5000))
tokenized_test = tokenized_test.select(range(1000))
print(f"✓ Using {len(tokenized_train)} train and {len(tokenized_test)} test samples")

# Set format for PyTorch
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load BERT model
print("\n[4/5] Initializing BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model.to(device)
print("✓ Model loaded")

# Metrics computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    report_to="none"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

# Train model
print("\n[5/5] Training model...")
print("-"*60)
trainer.train()

# Evaluate model
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1-Score: {eval_results['eval_f1_score']:.4f}")
print("="*60)

# Save model
print("\nSaving model to ./news_classifier...")
model.save_pretrained("./news_classifier")
tokenizer.save_pretrained("./news_classifier")
print("✓ Model saved successfully")

print("\n✓ Training complete! Run 'python app.py' to deploy.\n")