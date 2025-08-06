#!/usr/bin/env python3
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
    print(f"\n📋 Generated deck:")
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
        print(f"\n🃏 Sample mainboard cards:")
        for card in mainboard[:5]:
            print(f"   {card.get('quantity', 0)}x {card.get('name', 'Unknown')}")
    
    sideboard = deck.get('sideboard', [])
    if sideboard:
        print(f"\n🃏 Sample sideboard cards:")
        for card in sideboard[:3]:
            print(f"   {card.get('quantity', 0)}x {card.get('name', 'Unknown')}")
    
    # Check if local model was used
    generator_type = deck.get('generator', 'unknown')
    if 'local_finetuned' in generator_type:
        print("\n✅ SUCCESS: Local fine-tuned model is working!")
        return True
    elif 'api' in generator_type:
        print("\n⚠️  Using API fallback (local model not available)")
        print("   - Make sure you've trained the model with train_mtg_model.py")
        print("   - Check that ./mtg-deck-model/ directory exists")
        return True
    else:
        print("\n❌ FAILED: Unknown generator type")
        return False

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🎉 Integration test completed successfully!")
        print("\nNext steps:")
        print("1. Train your model: python train_mtg_model.py")
        print("2. Test your model: python test_mtg_model.py") 
        print("3. Use your chat app - it will automatically use the local model")
    else:
        print("\n❌ Integration test failed - check the errors above")
