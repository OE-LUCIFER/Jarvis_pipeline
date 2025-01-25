from dataclasses import dataclass
from typing import Dict, Optional
import json

@dataclass
class JarvisConfig:
    # Model Architecture
    vocab_size: int = 30000
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: int = 0.1
    attention_probs_dropout_prob: int = 0.1
    max_position_embeddings: int = 512
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Training Configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 10
    warmup_steps: int = 1000
    
    # Advanced Training Features
    use_mixed_precision: bool = False
    max_grad_norm: Optional[float] = None
    use_lr_scheduler: bool = False
    scheduler_type: str = "linear"  # linear, cosine, polynomial
    scheduler_warmup_steps: int = 0
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    use_weighted_loss: bool = False
    class_weights: Optional[Dict[str, float]] = None
    gradient_accumulation_steps: int = 1
    
    # Knowledge Distillation
    use_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 2.0
    alpha_distillation: float = 0.5  # Weight for distillation loss
    
    # Regularization
    label_smoothing: float = 0.0
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    
    # Categories
    categories = []
    
    @classmethod
    def from_json(cls, json_path: str) -> "JarvisConfig":
        """Load config from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        # Create a new instance
        config = cls()
        
        # Update all attributes from JSON
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle special cases for optional fields
        if config.class_weights is None and config.use_weighted_loss:
            config.class_weights = {cat: 1.0 for cat in config.categories}
            
        return config
    
    def save_to_json(self, json_path: str) -> None:
        """Save config to a JSON file."""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vars(self), f, indent=2)

    def __post_init__(self):
        if self.use_weighted_loss and self.class_weights is None:
            self.class_weights = {cat: 1.0 for cat in self.categories}
