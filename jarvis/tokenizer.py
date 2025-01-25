from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from typing import List, Union, Dict, Any
import json

class JarvisTokenizer:
    def __init__(self, vocab_size: int = 30000):
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
                ("[UNK]", 3),
                ("[PAD]", 0),
            ],
        )
        
        self.vocab_size = vocab_size
        
    def train(self, texts: List[str], output_path: str):
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"],
        )
        
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer.save(output_path)
        
    @classmethod
    def from_pretrained(cls, path: str):
        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_file(path)
        return tokenizer
        
    def save_pretrained(self, path: str):
        self.tokenizer.save(path)
        
    def encode(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ):
        if isinstance(text, str):
            text = [text]
            
        encoded = self.tokenizer.encode_batch(text)
        
        input_ids = []
        attention_mask = []
        
        for enc in encoded:
            if truncation and len(enc.ids) > max_length:
                ids = enc.ids[:max_length]
            else:
                ids = enc.ids
                
            if padding and len(ids) < max_length:
                pad_length = max_length - len(ids)
                ids = ids + [0] * pad_length
                mask = [1] * len(enc.ids) + [0] * pad_length
            else:
                mask = [1] * len(ids)
                
            input_ids.append(ids)
            attention_mask.append(mask)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def batch_encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: Single text or list of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary containing:
            - input_ids: List of token ID sequences
            - attention_mask: List of attention masks
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode all texts
        encoded_batch = [
            self.encode(text, max_length, padding, truncation)
            for text in texts
        ]
        
        # Combine results
        return {
            "input_ids": [e["input_ids"][0] for e in encoded_batch],
            "attention_mask": [e["attention_mask"][0] for e in encoded_batch]
        }
