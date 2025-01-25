import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os
from tqdm import tqdm
from .inference import JarvisClassifier

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.classifier = JarvisClassifier.from_pretrained(model_path)
        self.device = self.classifier.device
        
    def evaluate(
        self,
        test_data: List[Dict],
        batch_size: int = 32,
        output_dir: str = None
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: List of dictionaries containing text and label
            batch_size: Batch size for evaluation
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.classifier.model.eval()
        
        all_predictions = []
        all_labels = []
        all_scores = []
        
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch_data = test_data[i:i + batch_size]
            texts = [item["text"] for item in batch_data]
            labels = [item["label"] for item in batch_data]
            
            results = self.classifier.classify_with_scores(texts)
            predictions = [result[0] for result in results]
            scores = [result[1] for result in results]
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_scores.extend(scores)
        
        # Calculate metrics
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.classifier.config.categories,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(
            all_labels,
            all_predictions,
            labels=self.classifier.config.categories
        )
        
        # Calculate additional metrics
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        
        # Save results if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics
            metrics = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'classification_report': report
            }
            
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Plot and save confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.classifier.config.categories,
                yticklabels=self.classifier.config.categories
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
            
            # Save detailed predictions
            predictions_data = []
            for text, true_label, pred_label, scores in zip(
                [item["text"] for item in test_data],
                all_labels,
                all_predictions,
                all_scores
            ):
                predictions_data.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence_scores': scores,
                    'correct': true_label == pred_label
                })
            
            with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
    
    def analyze_errors(
        self,
        test_data: List[Dict],
        output_dir: str
    ) -> List[Dict]:
        """Analyze and categorize prediction errors."""
        results = []
        
        for item in tqdm(test_data, desc="Analyzing errors"):
            text = item["text"]
            true_label = item["label"]
            
            # Get prediction and confidence scores
            pred_label, scores = self.classifier.classify_with_scores(text)[0]
            
            if pred_label != true_label:
                # Get top 3 predictions
                top_3 = sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                results.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': scores[pred_label],
                    'top_3_predictions': top_3,
                    'error_type': self._categorize_error(
                        text, true_label, pred_label, scores
                    )
                })
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def _categorize_error(
        self,
        text: str,
        true_label: str,
        pred_label: str,
        scores: Dict[str, float]
    ) -> str:
        """Categorize the type of prediction error."""
        confidence = scores[pred_label]
        second_best = max(v for k, v in scores.items() if k != pred_label)
        
        if confidence < 0.3:
            return "low_confidence"
        elif confidence - second_best < 0.1:
            return "ambiguous"
        elif true_label in ['general', 'realtime'] and pred_label in ['general', 'realtime']:
            return "general_vs_realtime_confusion"
        elif true_label in ['open', 'close'] and pred_label in ['open', 'close']:
            return "action_confusion"
        else:
            return "major_misclassification"

# Example usage:
"""
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
"""
