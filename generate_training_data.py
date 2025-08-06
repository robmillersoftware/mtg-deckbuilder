#!/usr/bin/env python3
"""
Generate training data from real MTGTop8 meta data
"""

import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.card_database import CardDatabase
from data.metagame_analyzer import MetagameAnalyzer
from generation.deck_generator import CardEmbeddings

def generate_training_examples():
    print("🏗️  Generating Training Data from Real Meta Decks...")
    
    # Load card database and meta data
    card_db = CardDatabase("data/cards")
    if not card_db.load_latest_standard_cards():
        print("❌ No card database found")
        return
    
    analyzer = MetagameAnalyzer("data/raw", card_db)
    if not analyzer.load_scraped_data():
        print("❌ No meta data found")
        return
    
    print(f"✅ Loaded {len(card_db.standard_cards)} cards and {len(analyzer.deck_data)} decks")
    
    # Get archetype breakdown
    breakdown = analyzer.get_archetype_breakdown()
    print(f"✅ Found {len(breakdown)} unique archetypes")
    
    # Generate embeddings for training data
    print("🔄 Generating card embeddings for training...")
    embeddings = CardEmbeddings()
    card_list = list(card_db.standard_cards.values())
    embeddings.generate_card_embeddings(card_list)
    
    # Create training examples
    training_examples = []
    
    for archetype, stats in breakdown.items():
        if len(stats['sample_decks']) == 0:
            continue
        
        # Find a properly formatted deck (60 mainboard + 15 sideboard)
        valid_deck = None
        for sample_deck in stats['sample_decks']:
            if validate_deck_format(sample_deck):
                valid_deck = sample_deck
                break
        
        if not valid_deck:
            print(f"⚠️ Skipping {archetype} - no valid 75-card deck found")
            continue
            
        # Create training example
        training_example = create_training_example(
            valid_deck, archetype, stats, analyzer, embeddings, card_db
        )
        
        training_examples.append(training_example)
        
        print(f"📝 Created training example for {archetype} ({stats['percentage']:.1f}% of meta)")
    
    # Save training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.json"
    filepath = os.path.join("data/training", filename)
    
    os.makedirs("data/training", exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"\n✅ Generated {len(training_examples)} training examples")
    print(f"📁 Saved to: {filepath}")
    
    # Create human-readable version for editing
    create_editable_version(training_examples, timestamp)
    
    return filepath

def validate_deck_format(deck):
    """Validate that deck has exactly 60 mainboard + 15 sideboard cards"""
    mainboard = deck.get('mainboard', [])
    sideboard = deck.get('sideboard', [])
    
    mainboard_count = sum(card['quantity'] for card in mainboard)
    sideboard_count = sum(card['quantity'] for card in sideboard)
    
    return mainboard_count == 60 and sideboard_count == 15

def create_training_example(deck, archetype, stats, analyzer, embeddings, card_db):
    """Create a single training example from a deck"""
    
    # Basic deck info
    mainboard = deck.get('mainboard', [])
    sideboard = deck.get('sideboard', [])
    
    # Get deck colors
    colors = get_deck_colors(mainboard, card_db)
    
    # Get meta context
    meta_context = {
        'deck_to_beat': analyzer.get_deck_to_beat(),
        'top_cards': analyzer.get_top_cards(20, exclude_lands=True),
        'archetype_percentage': stats['percentage']
    }
    
    # Generate relevant card embeddings for this deck type
    relevant_cards = get_relevant_cards_for_deck(deck, embeddings, card_db)
    
    # Create the training example
    example = {
        'id': f"{archetype.lower().replace(' ', '_')}_{deck.get('event_name', 'unknown')}",
        'archetype': archetype,
        'meta_percentage': stats['percentage'],
        
        # Input data
        'input': {
            'prompt': f"Build a competitive {archetype} deck for the current Standard meta",
            'colors': colors,
            'archetype': archetype,
            'meta_context': meta_context,
            'available_cards': [card['name'] for card in relevant_cards[:100]]  # Top 100 relevant
        },
        
        # Output data (the actual deck)
        'output': {
            'mainboard': mainboard,
            'sideboard': sideboard,
            'total_cards': sum(card['quantity'] for card in mainboard),
            'colors': colors,
            'archetype': archetype,
        },
        
        # Placeholders for human annotation
        'strategy_explanation': f"[TO FILL: Explain why this {archetype} deck is effective against the current meta. What is the overall game plan?]",
        'key_card_choices': f"[TO FILL: Explain the key card choices and synergies in this deck. Why these specific cards?]",
        'meta_positioning': f"[TO FILL: How does this deck position against the meta leader ({meta_context['deck_to_beat']['name'] if meta_context['deck_to_beat'] else 'unknown'})?]",
        'sideboard_guide': f"[TO FILL: Explain the sideboard strategy and what matchups these cards target]",
        
        # Metadata
        'source': {
            'event': deck.get('event_name', 'Unknown'),
            'player': deck.get('player_name', 'Unknown'),
            'date': deck.get('event_date', 'Unknown'),
            'url': deck.get('event_url', '')
        }
    }
    
    return example

def get_deck_colors(mainboard, card_db):
    """Extract colors from deck mainboard"""
    colors = set()
    
    for card_entry in mainboard:
        card_name = card_entry['name']
        
        # Find card in database
        for card_id, card_data in card_db.standard_cards.items():
            if card_data.get('name', '').lower() == card_name.lower():
                card_colors = card_data.get('colors', [])
                colors.update(card_colors)
                break
    
    return list(colors)

def get_relevant_cards_for_deck(deck, embeddings, card_db, k=200):
    """Get cards that would be relevant for building this type of deck"""
    
    # Create a description of this deck for similarity search
    mainboard = deck.get('mainboard', [])
    
    # Sample key cards from the deck
    key_cards = [card['name'] for card in mainboard[:10]]  # First 10 cards
    
    # Create search query
    deck_description = f"deck with cards like {', '.join(key_cards[:5])}"
    
    # Find similar cards
    similar_cards = embeddings.find_similar_cards(deck_description, k=k)
    
    return [card for card, score in similar_cards]

def create_editable_version(training_examples, timestamp):
    """Create a human-readable version for editing"""
    
    filepath = os.path.join("data/training", f"training_examples_EDIT_ME_{timestamp}.md")
    
    with open(filepath, 'w') as f:
        f.write("# MTG Deck Training Examples - EDIT THIS FILE\n\n")
        f.write("Please fill in the strategy explanations, card choices, and meta positioning for each deck.\n\n")
        f.write("---\n\n")
        
        for i, example in enumerate(training_examples, 1):
            f.write(f"## Example {i}: {example['archetype']} ({example['meta_percentage']:.1f}% of meta)\n\n")
            
            # Deck list
            f.write("### Deck List:\n")
            f.write("**Mainboard:**\n")
            for card in example['output']['mainboard']:
                f.write(f"- {card['quantity']}x {card['name']}\n")
            
            f.write(f"\n**Sideboard:**\n")
            for card in example['output']['sideboard']:
                f.write(f"- {card['quantity']}x {card['name']}\n")
            
            f.write(f"\n**Colors:** {', '.join(example['output']['colors'])}\n")
            f.write(f"**Total Cards:** {example['output']['total_cards']}\n\n")
            
            # Fields to fill
            f.write("### PLEASE FILL IN:\n\n")
            f.write(f"**Strategy Explanation:**\n{example['strategy_explanation']}\n\n")
            f.write(f"**Key Card Choices:**\n{example['key_card_choices']}\n\n")
            f.write(f"**Meta Positioning:**\n{example['meta_positioning']}\n\n")
            f.write(f"**Sideboard Guide:**\n{example['sideboard_guide']}\n\n")
            
            # Meta context
            f.write("### Meta Context:\n")
            if example['input']['meta_context']['deck_to_beat']:
                deck_to_beat = example['input']['meta_context']['deck_to_beat']
                f.write(f"- **Current Meta Leader:** {deck_to_beat['name']} ({deck_to_beat['stats']['percentage']:.1f}%)\n")
            
            top_cards = [card for card, count in example['input']['meta_context']['top_cards'][:5]]
            f.write(f"- **Popular Cards:** {', '.join(top_cards)}\n")
            
            f.write(f"\n**Source:** {example['source']['event']} by {example['source']['player']}\n")
            f.write(f"**ID:** `{example['id']}`\n\n")
            f.write("---\n\n")
    
    print(f"📝 Created editable version: {filepath}")
    print("👆 Edit this file to add strategy explanations, then we can process it back into training data")

if __name__ == "__main__":
    generate_training_examples()