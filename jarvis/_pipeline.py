"""
Simple pipeline interface for JARVIS, similar to Hugging Face pipelines.
"""

from typing import Union, List, Dict, Any, Optional
import torch
from .inference import JarvisClassifier
from .config import JarvisConfig

class Pipeline:
    def __init__(
        self,
        task: str = "query-classification",
        model: Optional[str] = "jarvis-base",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a JARVIS pipeline.
        
        Args:
            task: Task to perform (currently only 'query-classification')
            model: Model name or path
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional arguments passed to the model
        """
        if task != "query-classification":
            raise ValueError(
                f"Task {task} not supported. "
                "Currently only 'query-classification' is supported."
            )
        
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load classifier
        self.classifier = JarvisClassifier.from_pretrained(model)
        
        # Move to device
        if self.device == "cuda":
            self.classifier.model.cuda()
        else:
            self.classifier.model.cpu()

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Create a pipeline from a pretrained model."""
        return cls(model=model_path, **kwargs)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_all: bool = True,
        threshold: float = 0.3,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Classify query text(s).
        
        Args:
            text: Input text or list of texts
            return_all: Return all labels above threshold
            threshold: Minimum confidence score (0.0 to 1.0)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary or list of dictionaries with:
            - labels: Predicted labels
            - scores: Confidence scores
            - embeddings: Text embeddings (if requested)
        """
        # Get classification results
        results = self.classifier.classify(
            text,
            return_all=return_all,
            threshold=threshold,
            include_embeddings=kwargs.get("include_embeddings", False),
            include_attention=kwargs.get("include_attention", False)
        )
        
        # Convert to simpler format
        if isinstance(text, str):
            return {
                "labels": results.labels,
                "scores": results.confidence_scores,
                "embeddings": results.embeddings
            }
        
        return [
            {
                "labels": r.labels,
                "scores": r.confidence_scores,
                "embeddings": r.embeddings
            }
            for r in results
        ]

def pipeline(
    task: str = "query-classification",
    model: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Create a JARVIS pipeline.
    
    Example:
        >>> pipe = pipeline("query-classification", model="models/jarvis-test")
        >>> result = pipe("open chrome and play music")
        >>> print(result["labels"])  # ['open', 'play']
    """
    return Pipeline(task=task, model=model, **kwargs)

# Add alias for backward compatibility
Pipe = pipeline
