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
    
    # Force CUDA usage if available
    device_map = None
    torch_dtype = torch.float32
    device = "cpu"
    
    if torch.cuda.is_available():
        print("🚀 Configuring for GPU training...")
        device = "cuda"
        device_map = "auto"
        torch_dtype = torch.float16  # Use FP16 for RTX 5090's efficiency
        
        # Set CUDA device explicitly and show memory info
        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        print(f"🚀 Using device {current_device}: {device_name}")
        print(f"🚀 Available GPU memory: {memory_gb:.1f} GB")
        print(f"🚀 PyTorch CUDA version: {torch.version.cuda}")
        
        # Force PyTorch to use CUDA
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    
    # Explicitly move to GPU if CUDA is available
    if device == "cuda":
        model = model.to(device)
        print(f"🚀 Model moved to GPU: {model.device}")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Model loaded. Parameters: {model.num_parameters():,}")
    
    # Verify GPU usage
    if torch.cuda.is_available():
        if hasattr(model, 'device'):
            print(f"🚀 Model device: {model.device}")
        else:
            # Check if model parameters are on GPU
            first_param_device = next(model.parameters()).device
            print(f"🚀 Model parameters device: {first_param_device}")
        
        # Show GPU memory usage
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        print(f"🚀 GPU memory: {allocated_memory:.2f} GB allocated, {cached_memory:.2f} GB cached")
    
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
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"🚀 GPU detected: {device_name}")
        print(f"🚀 GPU count: {device_count}")
        print(f"🚀 CUDA version: {torch.version.cuda}")
        print(f"🚀 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected - using CPU (will be slower)")
    
    print("🏗️  Setting up MTG Deck Generation Fine-tuning")
    print(f"Model: {MODEL_NAME}")
    
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
    
    # Setup training arguments optimized for RTX 5090
    gpu_available = torch.cuda.is_available()
    
    # Optimize batch size for RTX 5090 (32GB VRAM)
    if gpu_available:
        batch_size = 8  # Larger batch size for better GPU utilization
        gradient_accumulation = 2  # Reduce since we have bigger batches
        fp16_enabled = True
        dataloader_workers = 4
    else:
        batch_size = 2
        gradient_accumulation = 8
        fp16_enabled = False
        dataloader_workers = 0
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=100,
        logging_steps=5,  # More frequent logging
        eval_steps=25,    # More frequent evaluation
        save_steps=50,    # More frequent saving
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb logging
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=fp16_enabled,
        bf16=False,  # RTX 5090 supports bf16 but fp16 is more stable
        dataloader_pin_memory=gpu_available,
        dataloader_num_workers=dataloader_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        group_by_length=True,  # Optimize for variable-length sequences
        optim="adamw_torch",  # Use PyTorch's AdamW
        # Force GPU usage in training arguments
        no_cuda=False if gpu_available else True,
        use_cuda=gpu_available,
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
    
    # Final GPU check before training
    if torch.cuda.is_available():
        print(f"🚀 Final device check:")
        print(f"   - Model device: {next(model.parameters()).device}")
        print(f"   - CUDA current device: {torch.cuda.current_device()}")
        print(f"   - GPU memory before training: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Train the model
    print("🚀 Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if torch.cuda.is_available():
            print(f"GPU memory when failed: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        raise
    
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
