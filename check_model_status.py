#!/usr/bin/env python3
"""
Check the status of your local fine-tuned model
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_model_status():
    print("🔍 Checking local model status...")
    
    model_path = "./mtg-deck-model"
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print("\n📋 To train your model:")
        print("1. python train_mtg_model.py")
        return False
    
    print(f"✅ Model directory exists: {model_path}")
    
    # Check for required files
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files")
        print("Model may be incomplete - retrain with: python train_mtg_model.py")
        return False
    
    # Check if PEFT is available
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        print("✅ PEFT and transformers available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install peft transformers torch")
        return False
    
    # Try loading the model
    try:
        print("🤖 Testing model loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        # Don't actually load the full model in status check (too slow)
        print("✅ Model files appear valid")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    print("\n🎉 Model status: READY")
    print("\n📋 Your local model is ready to use!")
    print("Test with: python test_integration.py")
    return True

if __name__ == "__main__":
    check_model_status()
