import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, PolynomialLR
from tqdm import tqdm
import json
import os
from typing import List, Dict, Optional, Union
import numpy as np
from sklearn.metrics import classification_report

from .modeling import JarvisModel
from .tokenizer import JarvisTokenizer
from .config import JarvisConfig

class QueryDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: JarvisTokenizer, config: JarvisConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer.encode(
            item["query"], 
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation=True,
        )
        
        label_idx = self.config.categories.index(item["category"])
        
        return {
            "input_ids": torch.tensor(encoded["input_ids"][0], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"][0], dtype=torch.long),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

def get_scheduler(optimizer, config: JarvisConfig, num_training_steps: int):
    num_warmup_steps = config.scheduler_warmup_steps
    
    if config.scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=0
        )
    elif config.scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps - num_warmup_steps
        )
    elif config.scheduler_type == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=num_training_steps - num_warmup_steps,
            power=1.0
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

def compute_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0):
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

def train_model(
    data_path: str,
    output_dir: str,
    config: JarvisConfig = None,
    resume_from: str = None,
    val_data_path: Optional[str] = None,
):
    # Load and prepare data first to get categories
    with open(data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Get unique categories from data
    unique_categories = sorted(list(set(item["category"] for item in train_data)))
    
    # Initialize or update config with categories
    if config is None:
        config = JarvisConfig()
    config.categories = unique_categories
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load validation data if provided
    val_data = None
    if val_data_path:
        with open(val_data_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            
        # Update categories with validation data categories
        if val_data:
            val_categories = set(item["category"] for item in val_data)
            all_categories = sorted(list(set(unique_categories) | val_categories))
            config.categories = all_categories
    
    # Initialize tokenizer and train if needed
    if resume_from:
        tokenizer = JarvisTokenizer.from_pretrained(os.path.join(resume_from, "tokenizer.json"))
    else:
        tokenizer = JarvisTokenizer(vocab_size=config.vocab_size)
        tokenizer.train([item["query"] for item in train_data], os.path.join(output_dir, "tokenizer.json"))
    
    # Create datasets and dataloaders
    train_dataset = QueryDataset(train_data, tokenizer, config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = None
    if val_data:
        val_dataset = QueryDataset(val_data, tokenizer, config)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume_from:
        model = torch.load(os.path.join(resume_from, "model.pt"))
    else:
        model = JarvisModel(config)
    model.to(device)
    
    # Load teacher model if using knowledge distillation
    teacher_model = None
    if config.use_knowledge_distillation:
        teacher_model = torch.load(config.teacher_model_path)
        teacher_model.to(device)
        teacher_model.eval()
    
    # Initialize optimizer and loss
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize scheduler
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    if config.use_lr_scheduler:
        scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Initialize loss function
    if config.use_weighted_loss:
        class_weights = torch.tensor(
            [config.class_weights[cat] for cat in config.categories],
            device=device
        )
        loss_fn = CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    else:
        loss_fn = CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Initialize mixed precision training
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    ) if config.early_stopping else None
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(config.num_train_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Mixed precision training
            if config.use_mixed_precision:
                with autocast():
                    logits, _ = model(input_ids, attention_mask)
                    loss = loss_fn(logits, labels)
                    
                    if config.use_knowledge_distillation:
                        with torch.no_grad():
                            teacher_logits, _ = teacher_model(input_ids, attention_mask)
                        distill_loss = compute_kl_loss(
                            logits,
                            teacher_logits,
                            config.distillation_temperature
                        )
                        loss = (1 - config.alpha_distillation) * loss + config.alpha_distillation * distill_loss
                    
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if config.use_lr_scheduler:
                        scheduler.step()
            else:
                # Regular training
                logits, _ = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                
                if config.use_knowledge_distillation:
                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(input_ids, attention_mask)
                    distill_loss = compute_kl_loss(
                        logits,
                        teacher_logits,
                        config.distillation_temperature
                    )
                    loss = (1 - config.alpha_distillation) * loss + config.alpha_distillation * distill_loss
                
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if config.use_lr_scheduler:
                        scheduler.step()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Validation
        if val_dataloader:
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validating"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    logits, _ = model(input_ids, attention_mask)
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()
                    
                    val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(val_dataloader)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Print validation metrics
            report = classification_report(
                val_labels,
                val_preds,
                target_names=config.categories,
                zero_division=0
            )
            print("\nValidation Metrics:")
            print(report)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, os.path.join(output_dir, "best_model.pt"))
            
            # Early stopping
            if early_stopping and early_stopping(val_loss):
                print("Early stopping triggered")
                break
            
            model.train()
        
        # Save checkpoint
        torch.save(model, os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))
    
    # Save final model and config
    torch.save(model, os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"Training complete! Model saved to {output_dir}")
