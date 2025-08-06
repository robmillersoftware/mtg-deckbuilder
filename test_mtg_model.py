#!/usr/bin/env python3
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
    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    
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
        "Build a competitive Dimir Aggro deck for Standard format.\n\nContext: Archetype: Dimir Aggro | Colors: U, B | Meta percentage: 1.6% | Current meta leader: Mono Green (7.0%) | Popular cards in meta: Llanowar Elves, Mossborn Hydra, Sazh's Chocobo\n\nDeck:",
        
        "Build a competitive Mono Green deck for Standard format.\n\nContext: Archetype: Mono Green | Colors: G | Meta percentage: 7.0% | Popular cards in meta: Llanowar Elves, Mossborn Hydra, Tifa Lockhart\n\nDeck:",
        
        "Build a competitive Ur Soul Cauldron deck for Standard format.\n\nContext: Archetype: Ur Soul Cauldron | Colors: U, R | Meta percentage: 1.9% | Key available cards: Fear of Missing Out, Marauding Mako, Vivi Ornitier, Winternight Stories\n\nDeck:"
    ]
    
    print("🧪 Testing fine-tuned model...")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 Test {i}:")
        print(f"Prompt: {prompt[:100]}...")
        print("\n📝 Generated deck:")
        
        deck = generate_deck(model, tokenizer, prompt)
        print(deck)
        print("=" * 80)

if __name__ == "__main__":
    main()
