#!/usr/bin/env python3
"""
Test the full meta analysis through the chatbot interface
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interface.chatbot import MTGDeckbuildingChatbot

def test_full_meta():
    print("Testing Full Meta Analysis via Chatbot...")
    
    chatbot = MTGDeckbuildingChatbot()
    
    # Test the meta analysis
    response = chatbot._handle_meta_analysis("What's the current meta?")
    
    print("="*60)
    print(response)
    print("="*60)

if __name__ == "__main__":
    test_full_meta()