import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from data_cleaner import clean_text

def load_and_prepare_data(sample_size=1000):
    dataset = load_dataset("imdb")
    
    train_ds = dataset['train'].shuffle(seed=42).select(range(sample_size))
    test_ds = dataset['test'].shuffle(seed=42).select(range(sample_size // 5))
    
    def clean_batch(batch):
        batch['text'] = [clean_text(text) for text in batch['text']]
        return batch
        
    train_ds = train_ds.map(clean_batch, batched=True)
    test_ds = test_ds.map(clean_batch, batched=True)
    
    return train_ds, test_ds

def train_model(model_name="distilbert-base-uncased", output_dir=None):
    if output_dir is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(BASE_DIR, "models", "sentiment_model")

    train_ds, test_ds = load_and_prepare_data(sample_size=200)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
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
    
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train_model()
