#!/usr/bin/env python3
"""
Setup script for local fine-tuning of MTG deck generation model
"""

import json
import os
from typing import List, Dict, Any
import sys

def convert_training_data_to_hf_format(input_file: str, output_file: str):
    """Convert MTG training data to Hugging Face format"""
    
    print("🔄 Converting training data to Hugging Face format...")
    
    with open(input_file, 'r') as f:
        training_data = json.load(f)
    
    hf_examples = []
    
    for example in training_data:
        # Build the input prompt
        archetype = example['archetype']
        meta_percentage = example['meta_percentage']
        colors = example['output']['colors']
        
        # Get meta context if available
        meta_context = example['input'].get('meta_context', {})
        deck_to_beat = meta_context.get('deck_to_beat', {})
        top_cards = meta_context.get('top_cards', [])
        
        # Build context string
        context_parts = [
            f"Archetype: {archetype}",
            f"Colors: {', '.join(colors) if colors else 'Colorless'}",
            f"Meta percentage: {meta_percentage:.1f}%"
        ]
        
        if deck_to_beat and deck_to_beat.get('name'):
            context_parts.append(f"Current meta leader: {deck_to_beat['name']} ({deck_to_beat.get('stats', {}).get('percentage', 0):.1f}%)")
        
        if top_cards:
            popular_cards = [card for card, count in top_cards[:5]]
            context_parts.append(f"Popular cards in meta: {', '.join(popular_cards)}")
        
        # Available cards from the input
        available_cards = example['input'].get('available_cards', [])
        if available_cards:
            context_parts.append(f"Key available cards: {', '.join(available_cards[:10])}")
        
        context_str = " | ".join(context_parts)
        
        # Build the prompt
        prompt = f"Build a competitive {archetype} deck for Standard format.\n\nContext: {context_str}\n\nDeck:"
        
        # Format the output deck
        mainboard = example['output']['mainboard']
        sideboard = example['output']['sideboard']
        
        mainboard_str = "**Mainboard:**\n" + "\n".join([
            f"{card['quantity']}x {card['name']}" for card in mainboard
        ])
        
        sideboard_str = "**Sideboard:**\n" + "\n".join([
            f"{card['quantity']}x {card['name']}" for card in sideboard
        ])
        
        completion = f"{mainboard_str}\n\n{sideboard_str}"
        
        # Create training example
        hf_example = {
            "text": f"<|user|>\n{prompt}\n<|assistant|>\n{completion}\n<|end|>",
            "input": prompt,
            "output": completion,
            "archetype": archetype,
            "colors": colors,
            "meta_percentage": meta_percentage,
            "source": example.get('source', {}),
            "id": example.get('id', f"deck_{len(hf_examples)}")
        }
        
        hf_examples.append(hf_example)
    
    # Save the converted data
    with open(output_file, 'w') as f:
        json.dump(hf_examples, f, indent=2)
    
    print(f"✅ Converted {len(hf_examples)} training examples")
    print(f"📁 Saved to: {output_file}")
    
    return output_file

def create_training_script():
    """Create the main fine-tuning script"""
    
    script_content = '''#!/usr/bin/env python3
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
    print("\\nTo use your model:")
    print(f"from peft import AutoPeftModelForCausalLM")
    print(f"model = AutoPeftModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")

if __name__ == "__main__":
    main()
'''
    
    with open("train_mtg_model.py", "w") as f:
        f.write(script_content)
    
    print("📝 Created training script: train_mtg_model.py")

def create_inference_script():
    """Create a script for testing the trained model"""
    
    script_content = '''#!/usr/bin/env python3
"""
Inference script for testing the fine-tuned MTG deck model
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json

def load_model(model_path="./mtg-deck-model"):
    """Load the fine-tuned model"""
    print(f"🤖 Loading fine-tuned model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    return model, tokenizer

def generate_deck(model, tokenizer, prompt, max_length=800):
    """Generate a deck using the fine-tuned model"""
    
    # Format the prompt
    formatted_prompt = f"<|user|>\\n{prompt}\\n<|assistant|>\\n"
    
    # Tokenize
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|assistant|>" in generated_text:
        response = generated_text.split("<|assistant|>")[-1].strip()
        if "<|end|>" in response:
            response = response.split("<|end|>")[0].strip()
        return response
    
    return generated_text

def main():
    # Load model
    model, tokenizer = load_model()
    
    # Test prompts
    test_prompts = [
        "Build a competitive Dimir Aggro deck for Standard format.\\n\\nContext: Archetype: Dimir Aggro | Colors: U, B | Meta percentage: 1.6% | Current meta leader: Mono Green (7.0%) | Popular cards in meta: Llanowar Elves, Mossborn Hydra, Sazh's Chocobo\\n\\nDeck:",
        
        "Build a competitive Mono Green deck for Standard format.\\n\\nContext: Archetype: Mono Green | Colors: G | Meta percentage: 7.0% | Popular cards in meta: Llanowar Elves, Mossborn Hydra, Tifa Lockhart\\n\\nDeck:",
        
        "Build a competitive Ur Soul Cauldron deck for Standard format.\\n\\nContext: Archetype: Ur Soul Cauldron | Colors: U, R | Meta percentage: 1.9% | Key available cards: Fear of Missing Out, Marauding Mako, Vivi Ornitier, Winternight Stories\\n\\nDeck:"
    ]
    
    print("🧪 Testing fine-tuned model...")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\n🎯 Test {i}:")
        print(f"Prompt: {prompt[:100]}...")
        print("\\n📝 Generated deck:")
        
        deck = generate_deck(model, tokenizer, prompt)
        print(deck)
        print("=" * 80)

if __name__ == "__main__":
    main()
'''
    
    with open("test_mtg_model.py", "w") as f:
        f.write(script_content)
    
    print("📝 Created inference script: test_mtg_model.py")

def create_requirements_file():
    """Create requirements.txt for the project"""
    
    requirements = """# Core ML libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0

# Utilities
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.64.0

# Optional: For better performance
# bitsandbytes>=0.39.0  # For 8-bit training (if supported)
"""
    
    with open("requirements_finetuning.txt", "w") as f:
        f.write(requirements.strip())
    
    print("📝 Created requirements_finetuning.txt")

def create_setup_guide():
    """Create a setup guide markdown file"""
    
    guide_content = """# Local MTG Deck Fine-tuning Setup Guide

## Prerequisites

1. **Python 3.8+**
2. **GPU recommended** (NVIDIA with CUDA) but CPU works
3. **8GB+ RAM** (16GB+ recommended)
4. **5GB+ free disk space**

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements_finetuning.txt
```

### 2. Convert Training Data
```bash
python setup_local_finetuning.py
```
This converts your MTG training data to the format needed for fine-tuning.

### 3. Start Training
```bash
python train_mtg_model.py
```

### 4. Test Your Model
```bash
python test_mtg_model.py
```

## Model Options

Edit `MODEL_NAME` in `train_mtg_model.py`:

- `"microsoft/DialoGPT-medium"` - **Recommended** (117M params, good for dialogue)
- `"microsoft/DialoGPT-small"` - Faster training (117M params)
- `"gpt2"` - Classic choice (124M params)  
- `"EleutherAI/gpt-neo-125M"` - Slightly larger (125M params)

## Training Configuration

**Default settings** (adjust in `train_mtg_model.py`):
- **Epochs**: 3
- **Batch size**: 4 per device
- **Learning rate**: 5e-5
- **LoRA rank**: 16
- **Max sequence length**: 1024 tokens

## Expected Training Time

- **GPU (RTX 3080)**: ~30-60 minutes
- **CPU (8 cores)**: ~3-6 hours
- **Training examples**: 143 decks

## Monitoring Progress

Training will show:
```
Step 10/100: Loss: 2.345
Step 20/100: Loss: 2.123
...
```

Lower loss = better learning.

## Model Output Location

Trained model saved to: `./mtg-deck-model/`

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` from 4 to 2 or 1
- Reduce `max_length` from 1024 to 512
- Use `fp16=False` if on CPU

### Slow Training
- Use smaller model (`DialoGPT-small`)
- Reduce `num_train_epochs` from 3 to 1
- Increase `gradient_accumulation_steps`

### Poor Quality Output
- Increase training epochs
- Try different `temperature` values (0.3-1.0) during inference
- Add more training data

## Next Steps

1. **Test different prompts** with `test_mtg_model.py`
2. **Integrate into your main app** by loading the model
3. **Collect feedback** and retrain with better data
4. **Experiment with larger models** if results are promising

## Integration Example

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load your trained model
model = AutoPeftModelForCausalLM.from_pretrained("./mtg-deck-model")
tokenizer = AutoTokenizer.from_pretrained("./mtg-deck-model")

# Generate decks in your app
def generate_deck(archetype, colors):
    prompt = f"Build a competitive {archetype} deck for Standard format."
    # ... (rest of generation logic)
```
"""
    
    with open("FINETUNING_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("📝 Created setup guide: FINETUNING_GUIDE.md")

def main():
    """Main setup function"""
    print("🏗️  Setting up local fine-tuning environment...")
    
    # Find the latest training data file
    training_dir = "data/training"
    if os.path.exists(training_dir):
        training_files = [f for f in os.listdir(training_dir) if f.startswith("training_data_") and f.endswith(".json")]
        if training_files:
            latest_file = os.path.join(training_dir, sorted(training_files)[-1])
            print(f"📊 Found training data: {latest_file}")
            
            # Convert to HF format
            output_file = os.path.join(training_dir, "hf_training_data.json")
            convert_training_data_to_hf_format(latest_file, output_file)
        else:
            print("❌ No training data files found in data/training/")
            sys.exit(1)
    else:
        print("❌ Training data directory not found. Please run generate_training_data.py first.")
        sys.exit(1)
    
    # Create all the setup files
    create_training_script()
    create_inference_script()
    create_requirements_file()
    create_setup_guide()
    
    print("\n✅ Local fine-tuning setup complete!")
    print("\n📋 Next steps:")
    print("1. Transfer these files to your other machine:")
    print("   - train_mtg_model.py")
    print("   - test_mtg_model.py") 
    print("   - requirements_finetuning.txt")
    print("   - FINETUNING_GUIDE.md")
    print("   - data/training/hf_training_data.json")
    print("   - data/cards/ (card database)")
    print("\n2. Follow FINETUNING_GUIDE.md for setup instructions")
    print("\n🎯 Expected training time: 30-60 minutes on GPU, 3-6 hours on CPU")

if __name__ == "__main__":
    main()