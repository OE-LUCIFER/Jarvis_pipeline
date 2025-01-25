"""
JARVIS Query Classification Package
=================================

A powerful multilingual query classification system that intelligently categorizes
user queries into different types of actions and intents.
"""

from jarvis._pipeline import *
from jarvis.data_generator import MultilingualDataGenerator
from jarvis.inference import JarvisClassifier
from jarvis.train import train_model
from jarvis.config import JarvisConfig
from jarvis.evaluation import ModelEvaluator
from jarvis.modeling import JarvisModel
from jarvis.tokenizer import JarvisTokenizer

__version__ = "1.0.0"

# List of all supported languages
SUPPORTED_LANGUAGES = [
    # Indian Languages
    'hi',  # Hindi
    'bn',  # Bengali
    'te',  # Telugu
    'ta',  # Tamil
    'mr',  # Marathi
    'gu',  # Gujarati
    'kn',  # Kannada
    'ml',  # Malayalam
    'pa',  # Punjabi
    'or',  # Odia
    'ur',  # Urdu
    'sat', # Santali
    'kok', # Konkani
    'doi', # Dogri
    'mni', # Manipuri
    'as',  # Assamese
    'ks',  # Kashmiri
    'sd',  # Sindhi
    'mai', # Maithili
    'bho', # Bhojpuri
    'raj', # Rajasthani
    'ne',  # Nepali
    'si',  # Sinhala
    
    # International Languages
    'en',  # English
    'es',  # Spanish
    'fr',  # French
    'de',  # German
    'it',  # Italian
    'pt',  # Portuguese
    'ru',  # Russian
    'ar',  # Arabic
    'zh',  # Chinese
    'ja',  # Japanese
    'ko'   # Korean
]

# Query categories
QUERY_CATEGORIES = [
    'general',      # General knowledge queries
    'realtime',     # Realtime information queries
    'open',         # Open/launch commands
    'close',        # Close/exit commands
    'play',         # Media playback
    'generate_image', # Image generation
    'reminder',     # Reminders and alarms
    'system',       # System commands
    'content',      # Content generation
    'google_search', # Google search
    'youtube_search', # YouTube search
    'exit'          # Exit commands
]

__all__ = [
    'Pipeline',
    'MultilingualDataGenerator',
    'JarvisClassifier',
    'JarvisModel',
    'JarvisTokenizer',
    'JarvisConfig',
    'train_model',
    'ModelEvaluator',
    'SUPPORTED_LANGUAGES',
    'QUERY_CATEGORIES'
]