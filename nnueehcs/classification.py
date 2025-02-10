import torch
from torch import nn


class _IdOodClassifier:
    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        id_ipt, id_opt = id_data
        ood_ipt, ood_opt = ood_data
        model.eval()
        with torch.no_grad():
            id_preds, id_scores = model(id_ipt, return_ue=True)
            ood_preds, ood_scores = model(ood_ipt, return_ue=True)

        # Call _evaluate_scores to get the metrics
        metrics = self._evaluate_scores(id_scores, ood_scores)
        
        # Include predictions and scores in result
        metrics.update({
            "id_preds": id_preds,
            "ood_preds": ood_preds,
            "id_scores": id_scores,
            "ood_scores": ood_scores
        })
        
        return metrics


class PercentileBasedIdOodClassifier(_IdOodClassifier):
    def __init__(self, percentile: float):
        """Initialize classifier with percentile threshold
        
        Args:
            percentile: Percentile threshold (between 0 and 1)
            
        Raises:
            ValueError: If percentile is not between 0 and 1
        """
        if not 0 <= percentile <= 1:
            raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")
        super().__init__()
        self.percentile = percentile

    def _fpr(self, false_positives: int, true_negatives: int) -> float:
        """Calculate False Positive Rate
        
        Args:
            false_positives: Number of false positives
            true_negatives: Number of true negatives
            
        Returns:
            float: False Positive Rate
        """
        denominator = false_positives + true_negatives
        if denominator == 0:
            return 0.0
        return float(false_positives) / denominator

    def _fnr(self, false_negatives: int, true_positives: int) -> float:
        """Calculate False Negative Rate
        
        Args:
            false_negatives: Number of false negatives
            true_positives: Number of true positives
            
        Returns:
            float: False Negative Rate
        """
        denominator = false_negatives + true_positives
        if denominator == 0:
            return 0.0
        return float(false_negatives) / denominator

    def _sensitivity(self, true_positives: int, false_negatives: int) -> float:
        """Calculate Sensitivity (True Positive Rate)
        
        Args:
            true_positives: Number of true positives
            false_negatives: Number of false negatives
            
        Returns:
            float: Sensitivity
        """
        denominator = true_positives + false_negatives
        if denominator == 0:
            return 0.0
        return float(true_positives) / denominator

    def _specificity(self, true_negatives: int, false_positives: int) -> float:
        """Calculate Specificity (True Negative Rate)
        
        Args:
            true_negatives: Number of true negatives
            false_positives: Number of false positives
            
        Returns:
            float: Specificity
        """
        denominator = true_negatives + false_positives
        if denominator == 0:
            return 0.0
        return float(true_negatives) / denominator

    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        """Evaluate ID/OOD classification using percentile threshold
        
        Args:
            id_scores: Uncertainty scores for in-distribution data
            ood_scores: Uncertainty scores for out-of-distribution data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Ensure inputs are flattened
        id_scores = id_scores.reshape(-1)
        ood_scores = ood_scores.reshape(-1)
        
        # Calculate threshold based on ID scores percentile
        # For perfect separation case, we need to handle edge cases
        if torch.all(id_scores == id_scores[0]):
            # All ID scores are identical
            threshold = id_scores[0]
        else:
            threshold = torch.quantile(id_scores, self.percentile)
        
        # Count positives and negatives
        # By design, we want (1-percentile) of ID samples below threshold
        id_above = (id_scores > threshold).sum().item()
        id_below = (id_scores <= threshold).sum().item()
        ood_above = (ood_scores > threshold).sum().item()
        ood_below = (ood_scores <= threshold).sum().item()

        # Calculate metrics
        fpr = self._fpr(id_above, id_below)
        fnr = self._fnr(ood_below, ood_above)
        sensitivity = self._sensitivity(ood_above, ood_below)
        specificity = self._specificity(id_below, id_above)

        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr)
        }

    @classmethod
    def get_objectives(cls):
        return [{'name': 'sensitivity', 'type': 'maximize'}]

    @classmethod
    def get_metrics(cls):
        return ['sensitivity']


class ReversedPercentileBasedIdOodClassifier(PercentileBasedIdOodClassifier):
    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        """Evaluate with reversed logic (lower scores indicate OOD)
        
        Args:
            id_scores: Uncertainty scores for in-distribution data
            ood_scores: Uncertainty scores for out-of-distribution data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Ensure inputs are flattened
        id_scores = id_scores.reshape(-1)
        ood_scores = ood_scores.reshape(-1)
        
        # Use reversed percentile (1 - percentile)
        reverse_percentile = 1 - self.percentile
        id_percentile = torch.quantile(id_scores, reverse_percentile)
        
        # Count with reversed logic
        id_above = id_scores[id_scores > id_percentile]
        id_below = id_scores[id_scores <= id_percentile]

        ood_above = ood_scores[ood_scores > id_percentile]
        ood_below = ood_scores[ood_scores <= id_percentile]

        false_positives = len(id_below)
        false_negatives = len(ood_above)
        true_positives = len(ood_below)
        true_negatives = len(id_above)

        # Calculate metrics
        fpr = self._fpr(false_positives, true_negatives)
        fnr = self._fnr(false_negatives, true_positives)
        sensitivity = self._sensitivity(true_positives, false_negatives)
        specificity = self._specificity(true_negatives, false_positives)

        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr)
        }
