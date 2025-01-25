import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json
import os
from typing import List, Dict, Optional

from QueryMindClassifier.jarvis.inference import JarvisClassifier
from QueryMindClassifier.jarvis.train import QueryDataset
from .modeling import JarvisModel
from .tokenizer import JarvisTokenizer
from .config import JarvisConfig

class FineTuneConfig:
    def __init__(
        self,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        freeze_embeddings: bool = True,
        freeze_layers: Optional[List[int]] = None,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.freeze_embeddings = freeze_embeddings
        self.freeze_layers = freeze_layers or []

def freeze_model_parts(model: JarvisModel, config: FineTuneConfig):
    """Freeze specified parts of the model for fine-tuning."""
    if config.freeze_embeddings:
        for param in model.embeddings.parameters():
            param.requires_grad = False
            
    for layer_idx in config.freeze_layers:
        if layer_idx < len(model.encoder):
            for param in model.encoder[layer_idx].parameters():
                param.requires_grad = False

def finetune(
    base_model_path: str,
    train_data_path: str,
    output_dir: str,
    config: Optional[FineTuneConfig] = None,
    eval_data_path: Optional[str] = None,
):
    if config is None:
        config = FineTuneConfig()
        
    # Load base model and tokenizer
    model = JarvisClassifier.from_pretrained(base_model_path)
    freeze_model_parts(model.model, config)
    
    # Load training data
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Create dataset and dataloader
    train_dataset = QueryDataset(train_data, model.tokenizer, model.config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Load eval data if provided
    eval_dataloader = None
    if eval_data_path:
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        eval_dataset = QueryDataset(eval_data, model.tokenizer, model.config)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Initialize optimizer
    optimizer = AdamW(
        [p for p in model.model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    device = model.device
    model.model.train()
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _ = model.model(input_ids, attention_mask)
            loss = CrossEntropyLoss()(logits, labels)
            
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
                
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.model.parameters(),
                    config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluation
        if eval_dataloader:
            model.model.eval()
            eval_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    logits, _ = model.model(input_ids, attention_mask)
                    loss = CrossEntropyLoss()(logits, labels)
                    eval_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            avg_eval_loss = eval_loss / len(eval_dataloader)
            accuracy = correct / total
            print(f"Evaluation - Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            model.model.train()
        
        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.model, os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save(model.model, os.path.join(output_dir, "model.pt"))
    model.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer.json"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(model.config), f, indent=2)
    
    print(f"Fine-tuning complete! Model saved to {output_dir}")

# Example usage:
"""
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
"""
