# Jarvis

A powerful multilingual query classification system that intelligently categorizes user queries into different types of actions and intents. Built with advanced NLP capabilities and support for 25+ languages.

## 🌟 Features

<details>
  <summary>Click to expand</summary>
  
- **Multilingual Support**: 25+ languages including Hindi, Bengali, Telugu, Tamil, and more
- **Extensible Architecture**: Easy to add new languages and query types

</details>

<hr>

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

<details>
  <summary>Click to expand</summary>
  
```python
from jarvis import pipeline

# Initialize the pipeline
pipe = pipeline()

# Classify a query
result = pipe("open facebook")
print(result["labels"])  # Output: ['open']

# Multilingual query
result = pipe("नमस्ते, क्या हाल है?")
print(result["labels"])  # Output: ['general']

```
</details>

## 🔧 Advanced Usage

<details>
  <summary>Click to expand</summary>

### Custom Data Generation

```python
from jarvis import MultilingualDataGenerator

# Initialize generator
generator = MultilingualDataGenerator()

# Generate training data
data = generator.generate_data(
    num_samples=1000,
    languages=['en', 'hi', 'es'],
    output_file='training_data.json'
)
```

### Training From Scratch

```python
from jarvis.train import train_model
from jarvis.config import JarvisConfig

# Define training config
config = JarvisConfig(
    vocab_size=30000,
    num_train_epochs=3,
    batch_size=32,
    learning_rate=1e-4,
    use_mixed_precision=True,
    use_lr_scheduler=True,
    scheduler_warmup_steps=100,
    early_stopping=True,
    early_stopping_patience=2,
    gradient_accumulation_steps=2
)

# Train the model
train_model(
    data_path='training_data.json',
    output_dir='models/jarvis-trained',
    config=config
)
```

### Fine-tuning

```python
from jarvis.finetune import finetune, FineTuneConfig

# Basic fine-tuning
finetune(
    base_model_path="models/jarvis-base",
    train_data_path="data/finetune_data.json",
    output_dir="models/jarvis-finetuned"
)

# Advanced fine-tuning with custom configuration
config = FineTuneConfig(
    learning_rate=1e-5,
    num_epochs=5,
    batch_size=32,
    freeze_embeddings=True,
    freeze_layers=[0, 1, 2]  # Freeze first 3 layers
)

finetune(
    base_model_path="models/jarvis-base",
    train_data_path="data/finetune_data.json",
    output_dir="models/jarvis-finetuned",
    config=config,
    eval_data_path="data/eval_data.json"
)
```

### Evaluation

```python
from jarvis.evaluation import ModelEvaluator
import json

# Initialize evaluator
evaluator = ModelEvaluator("models/jarvis-base")

# Load test data
with open('data/test.json', 'r') as f:
    test_data = json.load(f)

# Run evaluation
metrics = evaluator.evaluate(
    test_data,
    batch_size=32,
    output_dir='evaluation_results'
)

# Print main metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

# Analyze errors
error_analysis = evaluator.analyze_errors(
    test_data,
    output_dir='evaluation_results'
)

# Print error distribution
error_types = [error['error_type'] for error in error_analysis]
for error_type in set(error_types):
    count = error_types.count(error_type)
    print(f"{error_type}: {count} errors")
```
</details>

## 🌍 Supported Languages

<details>
  <summary>Click to expand</summary>
  
1. **Indian Languages**
   - Hindi (hi)
   - Bengali (bn)
   - Telugu (te)
   - Tamil (ta)
   - Marathi (mr)
   - Gujarati (gu)
   - Kannada (kn)
   - Malayalam (ml)
   - Punjabi (pa)
   - Odia (or)
   - Urdu (ur)
   - Sanskrit (sa)
   - Santali (sat)
   - Konkani (kok)
   - Dogri (doi)
   - Manipuri (mni)
   - Assamese (as)
   - Kashmiri (ks)
   - Sindhi (sd)
   - Maithili (mai)

2. **International Languages**
   - English (en)
   - Spanish (es)
   - French (fr)
   - German (de)
   - Italian (it)
   - Portuguese (pt)
   - Russian (ru)
   - Arabic (ar)
   - Chinese (zh)
   - Japanese (ja)
   - Korean (ko)
</details>

<hr>

<div align="center">
  <a href="https://github.com/OE-LUCIFER/Jarvis_pipeline.git">
    <img src="https://img.shields.io/github/stars/OE-LUCIFER/Jarvis_pipeline?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/OE-LUCIFER/Jarvis_pipeline.git/fork">
    <img src="https://img.shields.io/github/forks/OE-LUCIFER/Jarvis_pipeline?style=social" alt="GitHub forks"/>
  </a>
  <a href="https://github.com/OE-LUCIFER/Jarvis_pipeline.git/issues">
    <img src="https://img.shields.io/github/issues/OE-LUCIFER/Jarvis_pipeline" alt="GitHub issues"/>
  </a>
</div>

<hr>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the HelpingAI License v3.0 - see the [LICENSE](LICENSE) file for details.
