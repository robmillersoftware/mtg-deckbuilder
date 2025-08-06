#!/usr/bin/env python3
"""
Local fine-tuning script for MTG deck generation
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from typing import Dict, List

def load_training_data(file_path: str):
    """Load and prepare training data"""
    print(f"📊 Loading training data from {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} training examples")
    
    # Extract just the text for training
    texts = [example['text'] for example in data]
    
    return Dataset.from_dict({"text": texts})

def setup_model_and_tokenizer(model_name: str = "microsoft/DialoGPT-medium"):
    """Setup the model and tokenizer"""
    print(f"🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer

def setup_lora_config():
    """Setup LoRA configuration for efficient fine-tuning"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
        bias="none",
        inference_mode=False,
    )

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the training examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )

def main():
    # Configuration
    MODEL_NAME = "microsoft/DialoGPT-medium"  # Good for dialogue/completion tasks
    TRAINING_DATA_FILE = "data/training/hf_training_data.json"
    OUTPUT_DIR = "./mtg-deck-model"
    MAX_LENGTH = 1024
    
    # Alternative models to try:
    # MODEL_NAME = "microsoft/DialoGPT-small"  # Faster training
    # MODEL_NAME = "gpt2"  # Classic choice
    # MODEL_NAME = "EleutherAI/gpt-neo-125M"  # Larger but still manageable
    
    print("🏗️  Setting up MTG Deck Generation Fine-tuning")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Load data
    dataset = load_training_data(TRAINING_DATA_FILE)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # Apply LoRA for efficient training
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize dataset
    print("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"📊 Training examples: {len(train_dataset)}")
    print(f"📊 Evaluation examples: {len(eval_dataset)}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb logging
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("🚀 Starting training...")
    trainer.train()
    
    # Save the final model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")
    print("\nTo use your model:")
    print(f"from peft import AutoPeftModelForCausalLM")
    print(f"model = AutoPeftModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")

if __name__ == "__main__":
    main()
