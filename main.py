#!/usr/bin/env python3
"""
MTG Deckbuilding AI System
Main entry point for the application
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interface.chatbot import main

if __name__ == "__main__":
    main()