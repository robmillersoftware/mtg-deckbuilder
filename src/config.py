import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Data Sources
    MTGTOP8_BASE_URL = "https://www.mtgtop8.com"
    SCRYFALL_API_BASE = "https://api.scryfall.com"
    MTGJSON_URL = "https://mtgjson.com/api/v5/Standard.json"
    
    # Paths
    DATA_DIR = "data"
    RAW_DATA_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
    CARDS_DATA_DIR = f"{DATA_DIR}/cards"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    
    # Model Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    
    # Simulation Settings
    SIMULATION_RUNS = 1000
    MAX_TURNS = 20
    
    # Web Interface
    HOST = "127.0.0.1"
    PORT = 7860
    DEBUG = True