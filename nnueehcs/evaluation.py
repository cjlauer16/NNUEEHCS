import torch
import torch.nn as nn
from typing import Union, Tuple
import numpy as np
from scipy.stats import wasserstein_distance
from abc import ABC, abstractmethod

class UncertaintyEstimate:
    def __init__(self, data: Union[np.ndarray, Tuple]):
        self.data = self._to_numpy(data)

    @property
    def dimensions(self) -> int:
        return len(self.data) if isinstance(self.data, tuple) else 1

    def flatten(self):
        assert self.dimensions == 1, "Can only flatten 1D uncertainty estimates"
        return self.data.flatten()

    def mean(self):
        return np.mean(self._combine())

    def _combine(self):
        if self.dimensions == 1:
            return self.data
        else:
            flat_dat = [d.flatten() for d in self.data]
            return np.concatenate(flat_dat)

    def _to_numpy(self, data: Union[np.ndarray, Tuple]) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, tuple):
            return tuple(self._to_numpy(d) for d in data)

class UncertaintyDistanceMetric(ABC):
    @abstractmethod
    def distance(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        pass

class WassersteinDistance(UncertaintyDistanceMetric):
    def distance(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        if ue1.dimensions != ue2.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")
        
        if ue1.dimensions == 1:
            return wasserstein_distance(ue1.flatten(), ue2.flatten())
        else:
            distances = [wasserstein_distance(ue1.data[i].flatten(), 
                                              ue2.data[i].flatten()) 
                         for i in range(ue1.dimensions)]
            return np.mean(distances)

class EuclideanDistance(UncertaintyDistanceMetric):
    def distance(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        if ue1.dimensions != ue2.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")
        
        return np.mean(np.sqrt(np.sum((ue1.data - ue2.data) ** 2, axis=-1)))

class UncertaintyEvaluator:
    def __init__(self, distance_metric: UncertaintyDistanceMetric):
        self.distance_metric = distance_metric

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        model.eval()
        id_ipt, id_opt = id_data
        ood_ipt, ood_opt = ood_data
        with torch.no_grad():
            id_preds, id_ue_raw = model(id_ipt, return_ue=True)
            ood_preds, ood_ue_raw = model(ood_ipt, return_ue=True)
        
        id_ue = UncertaintyEstimate(id_ue_raw)
        ood_ue = UncertaintyEstimate(ood_ue_raw)
        
        uncertainty_distance = self.distance_metric.distance(id_ue, ood_ue)

        loss_fn = model.val_loss
        
        return {
            "id_loss": loss_fn(id_preds, id_opt).item(),
            "ood_loss": loss_fn(ood_preds, ood_opt).item(),
            "avg_id_uncertainty": id_ue.mean().item(),
            "avg_ood_uncertainty": ood_ue.mean().item(),
            "uncertainty_distance": uncertainty_distance,
            "uncertainty_dimensions": id_ue.dimensions,
            'id_ue': id_ue,
            'ood_ue': ood_ue
        }

def get_uncertainty_evaluator(distance_metric: str) -> UncertaintyEvaluator:
    distance_metrics = {
        "wasserstein": WassersteinDistance,
        "euclidean": EuclideanDistance
    }
    
    metric = distance_metrics.get(distance_metric.lower())
    
    if not metric:
        raise ValueError("Invalid distance metric type")
    
    return UncertaintyEvaluator(metric())