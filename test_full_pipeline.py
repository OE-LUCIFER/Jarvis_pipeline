"""
Full pipeline test: Data Generation -> Training -> Inference
Tests the complete workflow with Hindi, Hinglish, and English.
"""

import os
import json
from jarvis import (
    MultilingualDataGenerator,
    train_model,
    JarvisConfig,
    Pipeline

)
from rich import print
def setup_directories():
    """Create necessary directories for data and models."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
def generate_training_data():
    """Generate training data in Hindi, Hinglish, and English."""
    print("\n1. Generating Training Data")
    print("-" * 50)
    
    generator = MultilingualDataGenerator()
    
    # Generate data for each language
    languages = ['en', 'hi']  # Hinglish will be handled in templates
    train_data = []
    
    # Basic queries in each language
    for lang in languages:
        data = generator.generate_data(
            num_samples=3000,
            languages=[lang]
        )
        train_data.extend(data)
    

    # Generate Hinglish data
    hinglish_data = generator.generate_data(
        num_samples=4000,
        languages=['hi-en'],

    )
    train_data.extend(hinglish_data)
    
    # Save training data
    with open("data/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(train_data)} total training samples")
    print("Sample entries:")
    for entry in train_data[:3]:
        print(json.dumps(entry, ensure_ascii=False, indent=2))

def train_classifier():
    """Train the classifier on the generated data."""
    print("\n2. Training Classifier")
    print("-" * 50)
    
    config = JarvisConfig(
        vocab_size=32000,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=6,
        max_position_embeddings=128,

    )
    
    train_model(
        data_path="data/train_data.json",
        output_dir="models/jarvis-test",
        config=config,

    )


def test_inference():
    """Test inference with various queries."""
    print("\n4. Testing Inference")
    print("-" * 50)

    pipe = Pipeline.from_pretrained("models/jarvis-test")
    
    test_queries = [
        # English queries
        "open chrome",
        "what is artificial intelligence?",
        "close facebook",
        "make an image of futuristic city",
        # Hinglish queries
        "chrome open karo",
        "facebook band karo",


    ]
    
    print("\nTesting various queries:")
    for query in test_queries:
        result = pipe(query)
        print(query)
        print(result["labels"])

def main():
    """Run the full pipeline test."""
    print("Starting Full Pipeline Test")
    print("=" * 50)
    
    # Create directories
    setup_directories()
    
    try:
        # Generate training data
        generate_training_data()
        
        # Train the model
        train_classifier()

        # Test inference
        test_inference()
        
        print("\nFull Pipeline Test Completed Successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
