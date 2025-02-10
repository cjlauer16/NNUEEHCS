import torch
import torch.nn as nn
from typing import Union, Tuple
import numpy as np
from scipy.stats import wasserstein_distance
from abc import ABC, abstractmethod
from .classification import PercentileBasedIdOodClassifier, ReversedPercentileBasedIdOodClassifier


class UncertaintyEstimate:
    def __init__(self, data: Union[np.ndarray, Tuple]):
        """Initialize UncertaintyEstimate with data
        
        Args:
            data: Input data as numpy array, torch tensor, or tuple of these
            
        Raises:
            ValueError: If data is empty or has invalid shape/values
        """
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.size == 0:
            raise ValueError("Cannot create UncertaintyEstimate from empty data")
        elif isinstance(data, tuple) and any(d.size == 0 for d in data):
            raise ValueError("Cannot create UncertaintyEstimate from empty tuple data")
            
        self.data = self._to_numpy(data)
        
        # Validate tuple data has matching dimensions
        if isinstance(self.data, tuple):
            shapes = [d.shape[0] for d in self.data]
            if len(set(shapes)) > 1:
                raise ValueError(f"All arrays in tuple must have same first dimension, got shapes: {shapes}")

    @property
    def dimensions(self) -> int:
        return len(self.data) if isinstance(self.data, tuple) else 1

    def flatten(self):
        if self.dimensions != 1:
            raise ValueError("Can only flatten 1D uncertainty estimates")
        return self.data.flatten()

    def mean(self):
        """Calculate mean of uncertainty estimate
        
        Returns:
            float: Mean value across all dimensions
            
        Note:
            Returns NaN if data contains NaN values
        """
        return np.mean(self._combine())

    def _combine(self):
        """Combine multi-dimensional data into single array
        
        Returns:
            np.ndarray: Combined data
            
        Raises:
            ValueError: If dimensions don't match for tuple data
        """
        if self.dimensions == 1:
            return self.data
        else:
            try:
                flat_dat = [d.flatten() for d in self.data]
                return np.concatenate(flat_dat)
            except ValueError as e:
                raise ValueError(f"Failed to combine data dimensions: {e}")

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, Tuple]) -> Union[np.ndarray, Tuple]:
        """Convert input data to numpy array(s)
        
        Args:
            data: Input data to convert
            
        Returns:
            Converted numpy array or tuple of arrays
            
        Raises:
            TypeError: If input data type is not supported
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, tuple):
            return tuple(self._to_numpy(d) for d in data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


class EvaluationMetric(ABC):
    """Base class for all evaluation metrics (both uncertainty and classification)"""
    @abstractmethod
    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        pass

    @classmethod
    @abstractmethod
    def get_objectives(cls):
        """Return list of objectives for optimization during training"""
        pass

    @classmethod
    @abstractmethod
    def get_metrics(cls):
        """Return list of all metrics this evaluator can compute"""
        pass


class UncertaintyEvaluationMetric(EvaluationMetric):
    """Base class for uncertainty evaluation metrics"""
    
    def evaluate(self, model, id_data: tuple, ood_data: tuple) -> dict:
        """Evaluate uncertainty estimates from model predictions
        
        Args:
            model: Model that returns uncertainty estimates
            id_data: Tuple of (inputs, outputs) for in-distribution data
            ood_data: Tuple of (inputs, outputs) for out-of-distribution data
            
        Returns:
            Dictionary containing evaluation metric(s)
        """
        model.eval()
        with torch.no_grad():
            _, id_scores = model(id_data[0], return_ue=True)
            _, ood_scores = model(ood_data[0], return_ue=True)
            
        id_ue = UncertaintyEstimate(id_scores)
        ood_ue = UncertaintyEstimate(ood_scores)
        
        result = self._evaluate_uncertainties(id_ue, ood_ue)
        
        # Ensure all values are Python floats
        return {k: float(v) for k, v in result.items()}

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        """Implement specific evaluation metric
        
        Args:
            id_ue: UncertaintyEstimate for in-distribution data
            ood_ue: UncertaintyEstimate for out-of-distribution data
            
        Returns:
            Dictionary containing evaluation metric(s)
        """
        raise NotImplementedError


class ClassificationMetric(EvaluationMetric):
    """Base class for classification-based metrics like TNR@TPR95, AUROC, etc."""
    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        with torch.no_grad():
            _, id_scores = model(id_data[0], return_ue=True)
            _, ood_scores = model(ood_data[0], return_ue=True)
        return self._evaluate_scores(id_scores, ood_scores)

    @abstractmethod
    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        pass


class WassersteinEvaluation(UncertaintyEvaluationMetric):
    name = "wasserstein_distance"

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        value = None

        if id_ue.dimensions == 1:
            value = wasserstein_distance(id_ue.flatten(), ood_ue.flatten())
        else:
            distances = [wasserstein_distance(id_ue.data[i].flatten(), 
                                              ood_ue.data[i].flatten()) 
                         for i in range(id_ue.dimensions)]
            value = np.mean(distances)
        return {self.name: value}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]


class EuclideanEvaluation(UncertaintyEvaluationMetric):
    name = "euclidean_distance"

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        result = np.mean(np.sqrt(np.sum((id_ue.data - ood_ue.data) ** 2, axis=-1)))
        return {self.name: float(result)}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]


class JensenShannonEvaluation(UncertaintyEvaluationMetric):
    name = "jensen_shannon_distance"

    def _to_probability_distribution(self, ue: UncertaintyEstimate) -> np.ndarray:
        if ue.dimensions == 1:
            return ue.data / np.sum(ue.data)
        else:
            return np.array([d / np.sum(d) for d in ue.data])

    def _is_probability_distribution(self, data: np.ndarray) -> bool:
        return np.allclose(np.sum(data), 1.0)

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        p1 = id_ue.data
        p2 = ood_ue.data

        result = self._average_js_distance(p1, p2)
        return {self.name: result}

    def _average_js_distance(self, array1: np.array, array2: np.array) -> float:
        from scipy.spatial.distance import jensenshannon
        p1 = array1
        p2 = array2

        if p1.ndim == 1 or (p1.ndim == 2 and p1.shape[1] == 1):
            p1flat = p1.flatten()
            p2flat = p2.flatten()
            # extend with zeros so their shapes match
            # js_distances = jensenshannon(p1flat, p2flat)
            return self.pdf_jsd(p1flat, p2flat)
        else:
            js_distances = [jensenshannon(p1[i], p2[i]) for i in range(p1.shape[0])]

        return np.mean(js_distances)

    def pdf_jsd(self, dist1, dist2, num_points=20000):
        from scipy.stats import gaussian_kde
        from scipy.spatial.distance import jensenshannon
        kde1 = gaussian_kde(dist1)
        kde2 = gaussian_kde(dist2)
        x_range = np.linspace(min(dist1.min(), dist2.min()), max(dist1.max(), dist2.max()), num_points)
        pdf1 = kde1(x_range)
        pdf2 = kde2(x_range)
        return jensenshannon(pdf1, pdf2)

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]


class TNRatTPX(ClassificationMetric):
    """Calculates True Negative Rate (TNR) at a specified True Positive Rate (TPR)"""
    def __init__(self, target_tpr: float, reversed: bool = False):
        """
        Args:
            target_tpr: The TPR level at which to calculate TNR (between 0 and 1)
        """
        if not 0 <= target_tpr <= 1:
            raise ValueError(f"target_tpr must be between 0 and 1, got {target_tpr}")
        self.target_tpr = target_tpr
        # Create metric name based on percentage (e.g., 'tnr_at_tpr95' for 0.95)
        self.metric_name = f'tnr_at_tpr'
        self.reversed = reversed
    @classmethod
    def from_config(cls, config: dict) -> 'TNRatTPX':
        """Factory method to create from config dictionary"""
        return cls(target_tpr=config['target_tpr'], reversed=config.get('reversed', False))

    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        # Flatten scores if they're multi-dimensional
        id_scores = id_scores.reshape(-1)
        ood_scores = ood_scores.reshape(-1)
        
        # Sort scores to compute TPR and TNR
        thresholds = torch.sort(torch.cat([id_scores, ood_scores])).values
        
        # Calculate TPR and TNR for each threshold
        tpr_values = []
        tnr_values = []
        
        for threshold in thresholds:
            # For OOD detection:
            # True Positive: OOD correctly identified as OOD (score > threshold)
            # True Negative: ID correctly identified as ID (score <= threshold)
            if self.reversed:
                tp = (id_scores > threshold).sum().item()
                tn = (ood_scores <= threshold).sum().item()
                fp = (id_scores <= threshold).sum().item()
                fn = (ood_scores > threshold).sum().item()
            else:
                tp = (ood_scores > threshold).sum().item()
                tn = (id_scores <= threshold).sum().item()
                fp = (id_scores > threshold).sum().item()
                fn = (ood_scores <= threshold).sum().item()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            tpr_values.append(tpr)
            tnr_values.append(tnr)
        
        # Convert to numpy for interpolation
        tpr_values = torch.tensor(tpr_values)
        tnr_values = torch.tensor(tnr_values)
        
        # Find the TNR at the target TPR
        # Get the closest TPR value that's >= our target
        mask = tpr_values >= self.target_tpr
        if not mask.any():
            tnr_at_tpr = 0.0  # If we never reach the target TPR
        else:
            idx = mask.nonzero()[0][0]  # First index where TPR >= target
            tnr_at_tpr = tnr_values[idx].item()

        return {self.metric_name: tnr_at_tpr}

    @classmethod
    def get_objectives(cls):
        # Note: This is a class method, so we can't access self.metric_name
        # Instead, the actual metric name will be set when instantiating
        return [{'name': 'tnr_at_tpr', 'type': 'maximize'}]

    @classmethod
    def get_metrics(cls):
        # Similar to get_objectives, actual name set during instantiation
        return ['tnr_at_tpr']

    def get_instance_objectives(self):
        """Instance-specific objectives with correct metric name"""
        return [{'name': self.metric_name, 'type': 'maximize'}]

    def get_instance_metrics(self):
        """Instance-specific metrics with correct metric name"""
        return [self.metric_name]


class MetricEvaluator:
    """Unified evaluator that can handle multiple metrics"""
    def __init__(self, metrics: list[EvaluationMetric]):
        self.metrics = metrics

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        results = {}
        for metric in self.metrics:
            results.update(metric.evaluate(model, id_data, ood_data))
        return results

    def get_training_objectives(self):
        """Get objectives for optimization during training"""
        objectives = []
        for metric in self.metrics:
            # Use instance-specific objectives if available
            if hasattr(metric, 'get_instance_objectives'):
                objectives.extend(metric.get_instance_objectives())
            else:
                objectives.extend(metric.get_objectives())
        return objectives

    def get_all_metrics(self):
        """Get all available metrics for post-hoc analysis"""
        metrics = []
        for metric in self.metrics:
            # Use instance-specific metrics if available
            if hasattr(metric, 'get_instance_metrics'):
                metrics.extend(metric.get_instance_metrics())
            else:
                metrics.extend(metric.get_metrics())
        return metrics


def get_evaluator(config: dict) -> MetricEvaluator:
    """Factory function to create evaluator from config"""
    metrics = []
    for metric_config in config['metrics']:
        metric_type = metric_config['type']
        if metric_type == 'wasserstein':
            metrics.append(WassersteinEvaluation())
        elif metric_type == 'percentile_classification':
            if metric_config.get('reversed', False):
                metrics.append(ReversedPercentileBasedIdOodClassifier(metric_config['threshold']))
            else:
                metrics.append(PercentileBasedIdOodClassifier(metric_config['threshold']))
        elif metric_type == 'tnr_at_tpr':
            metrics.append(TNRatTPX.from_config(metric_config))
        # Add other metric types as needed
    
    return MetricEvaluator(metrics)


def get_uncertainty_evaluator(metric_config: str | dict) -> EvaluationMetric:
    """Factory function to create evaluator from config
    
    Args:
        metric_config: Either a string naming the metric or a dict with metric configuration
            If dict, must contain 'name' key and any required parameters for that metric
    
    Returns:
        Configured evaluation metric
    """
    # Handle string input for backward compatibility
    if isinstance(metric_config, str):
        metric_config = {'name': metric_config}

    distance_metrics = {
        WassersteinEvaluation.name: WassersteinEvaluation,
        EuclideanEvaluation.name: EuclideanEvaluation,
        JensenShannonEvaluation.name: JensenShannonEvaluation
    }

    name = metric_config['name']
    
    # Handle distance-based metrics
    if name in distance_metrics:
        return distance_metrics[name]()
    
    # Handle classification-based metrics
    if name == 'percentile_classification':
        threshold = metric_config['threshold']
        is_reversed = metric_config.get('reversed', False)
        return (ReversedPercentileBasedIdOodClassifier if is_reversed 
                else PercentileBasedIdOodClassifier)(threshold)
    elif name == 'tnr_at_tpr':
        target_tpr = metric_config['target_tpr']
        reversed = metric_config.get('reversed', False)
        return TNRatTPX(target_tpr, reversed)
            
    raise ValueError(f"Invalid metric type: {name}")
