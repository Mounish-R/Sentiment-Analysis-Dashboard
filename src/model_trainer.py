import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from data_cleaner import clean_text

def load_and_prepare_data(sample_size=1000):
    """
    Load a sample sentiment dataset (IMDB) and prepare it for training.
    You can replace this with your custom dataset loading logic.
    """
    print(f"Loading IMDb dataset (subset: {sample_size} examples)...")
    dataset = load_dataset("imdb")
    
    # Take a small subset for demonstration/speed
    train_ds = dataset['train'].shuffle(seed=42).select(range(sample_size))
    test_ds = dataset['test'].shuffle(seed=42).select(range(sample_size // 5))
    
    # Clean text
    print("Cleaning text data...")
    def clean_batch(batch):
        batch['text'] = [clean_text(text) for text in batch['text']]
        return batch
        
    train_ds = train_ds.map(clean_batch, batched=True)
    test_ds = test_ds.map(clean_batch, batched=True)
    
    return train_ds, test_ds

def train_model(model_name="distilbert-base-uncased", output_dir=None):
    """
    Fine-tune a Hugging Face transformer model for sentiment analysis.
    """
    if output_dir is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(BASE_DIR, "models", "sentiment_model")

    train_ds, test_ds = load_and_prepare_data(sample_size=100)
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    
    print(f"Loading model: {model_name}")
    # 2 labels: negative (0) and positive (1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train_model()
