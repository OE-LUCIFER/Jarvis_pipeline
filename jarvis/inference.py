import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from .modeling import JarvisModel
from .tokenizer import JarvisTokenizer
from .config import JarvisConfig

@dataclass
class ClassificationResult:
    """Holds the classification result with confidence scores and embeddings."""
    labels: List[str]
    confidence_scores: Dict[str, float]
    embeddings: Optional[np.ndarray] = None
    attention_weights: Optional[List[np.ndarray]] = None

    def __str__(self) -> str:
        scores_str = "\n".join(
            f"  {label}: {score:.4f}"
            for label, score in sorted(
                self.confidence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Show top 3 scores
        )
        return f"Labels: {self.labels}\nTop confidence scores:\n{scores_str}"

class JarvisClassifier:
    def __init__(
        self,
        model: JarvisModel,
        tokenizer: JarvisTokenizer,
        config: JarvisConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "JarvisClassifier":
        """Load a pretrained model from disk."""
        model = torch.load(f"{model_path}/model.pt", map_location="cpu")
        tokenizer = JarvisTokenizer.from_pretrained(f"{model_path}/tokenizer.json")
        config = JarvisConfig.from_json(f"{model_path}/config.json")
        return cls(model, tokenizer, config)

    def get_embeddings(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Get embeddings for input text(s)."""
        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer.batch_encode(
            text,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            input_ids = torch.tensor(encoded["input_ids"], device=self.device)
            attention_mask = torch.tensor(encoded["attention_mask"], device=self.device)
            
            _, embeddings = self.model(input_ids, attention_mask)
            
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()

    def compute_similarity(
        self,
        texts1: Union[str, List[str]],
        texts2: Union[str, List[str]]
    ) -> Union[float, np.ndarray]:
        """Compute cosine similarity between texts."""
        emb1 = self.get_embeddings(texts1)
        emb2 = self.get_embeddings(texts2)
        
        similarities = np.dot(emb1, emb2.T)
        
        if isinstance(texts1, str) and isinstance(texts2, str):
            return float(similarities[0, 0])
        return similarities

    def classify(
        self,
        texts: Union[str, List[str]],
        return_all: bool = True,
        threshold: float = 0.3,
        include_embeddings: bool = False,
        include_attention: bool = False,
    ) -> Union[ClassificationResult, List[ClassificationResult]]:
        """Classify text(s)."""
        # Encode texts
        encoded = self.tokenizer.batch_encode(
            texts,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation=True
        )
        
        # Convert to tensors
        input_ids = torch.tensor(encoded["input_ids"]).to(self.device)
        attention_mask = torch.tensor(encoded["attention_mask"]).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=include_attention
            )
            
            if include_attention:
                logits, embeddings, attention_weights = outputs
            else:
                logits, embeddings = outputs
            
            probs = torch.softmax(logits, dim=-1)
        
        # Process single text
        if isinstance(texts, str):
            return self._process_single_prediction(
                probs[0],
                embeddings[0] if include_embeddings else None,
                attention_weights[0] if include_attention else None,
                return_all,
                threshold
            )
            
        # Process batch of texts
        return [
            self._process_single_prediction(
                p,
                e if include_embeddings else None,
                a[0] if include_attention else None,
                return_all,
                threshold
            )
            for p, e, a in zip(
                probs,
                embeddings if include_embeddings else [None] * len(texts),
                attention_weights if include_attention else [None] * len(texts)
            )
        ]

    def _process_single_prediction(
        self,
        probs: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        attention_weights: Optional[List[torch.Tensor]] = None,
        return_all: bool = True,
        threshold: float = 0.3
    ) -> ClassificationResult:
        """
        Process a single prediction output.
        
        Args:
            probs: Probability distribution over classes
            embeddings: Text embeddings (optional)
            attention_weights: Attention weights from transformer layers (optional)
            return_all: Whether to return all labels above threshold
            threshold: Confidence score threshold
        
        Returns:
            ClassificationResult object containing predictions
        """
        # Convert to numpy for processing
        probs_np = probs.cpu().numpy()
        
        # Get confidence scores for all categories
        confidence_scores = {
            cat: float(prob)
            for cat, prob in zip(self.config.categories, probs_np)
        }
        
        if return_all:
            # Get all labels above threshold
            labels = [
                cat for cat, score in confidence_scores.items()
                if score >= threshold
            ]
            if not labels:  # If no label above threshold, return highest scoring
                labels = [max(confidence_scores.items(), key=lambda x: x[1])[0]]
        else:
            # Get only the highest scoring label
            labels = [max(confidence_scores.items(), key=lambda x: x[1])[0]]
        
        # Process embeddings if present
        if embeddings is not None:
            embeddings = embeddings.cpu().numpy()
            
        # Process attention weights if present
        if attention_weights is not None:
            attention_weights = [
                attn.cpu().numpy() for attn in attention_weights
            ]
        
        return ClassificationResult(
            labels=labels,
            confidence_scores=confidence_scores,
            embeddings=embeddings,
            attention_weights=attention_weights
        )

    def classify_with_explanation(
        self,
        text: str,
        num_samples: int = 10
    ) -> Tuple[ClassificationResult, List[Dict[str, float]]]:
        """
        Classify text and provide explanation through attention analysis.
        
        Args:
            text: Input text to classify
            num_samples: Number of similar examples to return
        
        Returns:
            Tuple of (ClassificationResult, explanation)
        """
        # Get classification with attention weights
        result = self.classify(
            text,
            include_attention=True,
            include_embeddings=True
        )
        
        # Analyze attention patterns
        words = self.tokenizer.tokenize(text)
        attention_scores = np.mean([layer_attn.mean(axis=0) for layer_attn in result.attention_weights], axis=0)
        
        # Get word importance scores
        word_scores = {
            word: float(score)
            for word, score in zip(words, attention_scores)
        }
        
        return result, word_scores

    def find_similar_queries(
        self,
        query: str,
        examples: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar queries from a list of examples."""
        query_emb = self.get_embeddings(query)
        example_embs = self.get_embeddings(examples)
        
        similarities = np.dot(query_emb, example_embs.T)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            (examples[idx], float(similarities[idx]))
            for idx in top_indices
        ]
