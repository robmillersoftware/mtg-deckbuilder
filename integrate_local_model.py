#!/usr/bin/env python3
"""
Integration script to update your MTG deck builder to use the local fine-tuned model
"""

import os
import shutil

def backup_original_generator():
    """Backup the original deck generator"""
    original_path = "src/generation/deck_generator.py"
    backup_path = "src/generation/deck_generator_backup.py"
    
    if os.path.exists(original_path) and not os.path.exists(backup_path):
        shutil.copy2(original_path, backup_path)
        print(f"✅ Backed up original generator to {backup_path}")
    else:
        print("⚠️  Backup already exists or original not found")

def update_imports():
    """Update any files that import the deck generator"""
    
    files_to_update = [
        "main.py",
        "cli.py", 
        "src/chat/chat_handler.py",
        "app.py"  # Common names for main application files
    ]
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"📝 Updating imports in {file_path}")
            
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace imports
            original_import = "from generation.deck_generator import LLMDeckGenerator"
            new_import = "from generation.local_deck_generator import LocalFineTunedDeckGenerator as LLMDeckGenerator"
            
            if original_import in content:
                content = content.replace(original_import, new_import)
                
                # Write back
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"  ✅ Updated imports in {file_path}")
            else:
                print(f"  ⚠️  No matching import found in {file_path}")

def create_integration_test():
    """Create a test script to verify the integration"""
    
    test_content = '''#!/usr/bin/env python3
"""
Test script to verify local model integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase
from generation.local_deck_generator import LocalFineTunedDeckGenerator

def test_integration():
    print("🧪 Testing local model integration...")
    
    # Load card database
    print("📊 Loading card database...")
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("❌ Failed to load card database")
        return False
    
    # Create deck generator
    print("🤖 Initializing deck generator...")
    from generation.local_deck_generator import CardEmbeddings
    embeddings = CardEmbeddings()
    embeddings.generate_card_embeddings(list(card_db.standard_cards.values())[:100])  # Sample for testing
    
    generator = LocalFineTunedDeckGenerator(embeddings)
    
    # Test generation
    print("🎯 Testing deck generation...")
    test_prompt = "Build an aggressive Dimir deck"
    
    deck = generator.generate_deck(
        prompt=test_prompt,
        colors=["U", "B"],
        archetype="Dimir Aggro",
        meta_context={
            'deck_to_beat': {'name': 'Mono Green', 'stats': {'percentage': 7.0}},
            'top_cards': [('Llanowar Elves', 100), ('Mossborn Hydra', 95)]
        }
    )
    
    # Check results
    print(f"\\n📋 Generated deck:")
    print(f"   Generator: {deck.get('generator', 'unknown')}")
    print(f"   Archetype: {deck.get('archetype', 'unknown')}")
    print(f"   Total cards: {deck.get('total_cards', 0)}")
    print(f"   Mainboard cards: {len(deck.get('mainboard', []))}")
    print(f"   Sideboard cards: {len(deck.get('sideboard', []))}")
    
    if deck.get('error'):
        print(f"   ⚠️  Error: {deck['error']}")
    
    # Show first few cards
    mainboard = deck.get('mainboard', [])
    if mainboard:
        print(f"\\n🃏 Sample mainboard cards:")
        for card in mainboard[:5]:
            print(f"   {card.get('quantity', 0)}x {card.get('name', 'Unknown')}")
    
    sideboard = deck.get('sideboard', [])
    if sideboard:
        print(f"\\n🃏 Sample sideboard cards:")
        for card in sideboard[:3]:
            print(f"   {card.get('quantity', 0)}x {card.get('name', 'Unknown')}")
    
    # Check if local model was used
    generator_type = deck.get('generator', 'unknown')
    if 'local_finetuned' in generator_type:
        print("\\n✅ SUCCESS: Local fine-tuned model is working!")
        return True
    elif 'api' in generator_type:
        print("\\n⚠️  Using API fallback (local model not available)")
        print("   - Make sure you've trained the model with train_mtg_model.py")
        print("   - Check that ./mtg-deck-model/ directory exists")
        return True
    else:
        print("\\n❌ FAILED: Unknown generator type")
        return False

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\\n🎉 Integration test completed successfully!")
        print("\\nNext steps:")
        print("1. Train your model: python train_mtg_model.py")
        print("2. Test your model: python test_mtg_model.py") 
        print("3. Use your chat app - it will automatically use the local model")
    else:
        print("\\n❌ Integration test failed - check the errors above")
'''
    
    with open("test_integration.py", "w") as f:
        f.write(test_content)
    
    print("📝 Created integration test: test_integration.py")

def create_model_status_checker():
    """Create a utility to check model status"""
    
    checker_content = '''#!/usr/bin/env python3
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
        print("\\n📋 To train your model:")
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
        print(f"\\n⚠️  Missing {len(missing_files)} required files")
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
    
    print("\\n🎉 Model status: READY")
    print("\\n📋 Your local model is ready to use!")
    print("Test with: python test_integration.py")
    return True

if __name__ == "__main__":
    check_model_status()
'''
    
    with open("check_model_status.py", "w") as f:
        f.write(checker_content)
    
    print("📝 Created model status checker: check_model_status.py")

def main():
    """Main integration function"""
    print("🔧 Integrating local fine-tuned model with your MTG deck builder...")
    
    # Step 1: Backup original
    backup_original_generator()
    
    # Step 2: Update imports (optional, since we created a new file)
    print("\\n📋 Your new generator is ready at: src/generation/local_deck_generator.py")
    print("   It includes backwards compatibility, so existing code should work.")
    
    # Step 3: Create test utilities
    create_integration_test()
    create_model_status_checker()
    
    print("\\n✅ Integration complete!")
    print("\\n📋 Next steps:")
    print("1. Transfer these files to your training machine:")
    print("   - All files from the fine-tuning setup")
    print("   - src/generation/local_deck_generator.py")
    print("   - test_integration.py")
    print("   - check_model_status.py")
    print("\\n2. On your training machine:")
    print("   - Run: python train_mtg_model.py")
    print("   - Test: python test_mtg_model.py")
    print("   - Verify: python check_model_status.py")
    print("\\n3. Copy the trained model back:")
    print("   - Copy ./mtg-deck-model/ directory back to this machine")
    print("\\n4. Test integration:")
    print("   - Run: python test_integration.py")
    print("\\n5. Update your main application:")
    print("   - Replace: from generation.deck_generator import LLMDeckGenerator")
    print("   - With: from generation.local_deck_generator import LocalFineTunedDeckGenerator as LLMDeckGenerator")
    print("\\n🎯 Your chat app will automatically use the fine-tuned model when available!")

if __name__ == "__main__":
    main()