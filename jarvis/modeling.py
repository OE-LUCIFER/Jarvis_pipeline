import torch
import torch.nn as nn
from typing import Optional, Tuple

class JarvisAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class JarvisLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = JarvisAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layernorm1(attention_output + hidden_states)
        
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(layer_output + attention_output)
        
        return layer_output

class JarvisModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.encoder = nn.ModuleList([JarvisLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, len(config.categories))
        
        self.init_weights()
        
    def init_weights(self):
        self.embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = embeddings + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        hidden_states = embeddings
        all_attentions = []
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)
            if output_attentions:
                all_attentions.append(hidden_states)
        
        pooled_output = self.pooler(hidden_states[:, 0])
        logits = self.classifier(pooled_output)
        
        if output_attentions:
            return logits, hidden_states, all_attentions
        
        return logits, hidden_states
