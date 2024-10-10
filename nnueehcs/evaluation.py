import torch
import torch.nn as nn
from typing import Union, Tuple
import numpy as np
from scipy.stats import wasserstein_distance
from abc import ABC, abstractmethod
from .classification import PercentileBasedIdOodClassifier


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


class UncertaintyEvaluationResult:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def get_result(self):
        return {self._name: self._value}

    def get_all_results(self):
        return self.get_result()


class ClassificationEvaluationResult:
    def __init__(self, metric_names, data):
        self.metric_names = metric_names
        self.data = data

    def get_result(self):
        return {name: self.data[name] for name in self.metric_names}

    def get_all_results(self):
        return self.data


class UncertaintyEvaluationMetric(ABC):
    @abstractmethod
    def evaluate(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        pass

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]


class WassersteinEvaluation(UncertaintyEvaluationMetric):
    name = "wasserstein_distance"

    def evaluate(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        if ue1.dimensions != ue2.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        value = None

        if ue1.dimensions == 1:
            value = wasserstein_distance(ue1.flatten(), ue2.flatten())
        else:
            distances = [wasserstein_distance(ue1.data[i].flatten(), 
                                              ue2.data[i].flatten()) 
                         for i in range(ue1.dimensions)]
            value = np.mean(distances)
        return UncertaintyEvaluationResult(WassersteinEvaluation.name, value)


class EuclideanEvaluation(UncertaintyEvaluationMetric):
    name = "euclidean_distance"

    def evaluate(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        if ue1.dimensions != ue2.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        result = np.mean(np.sqrt(np.sum((ue1.data - ue2.data) ** 2, axis=-1)))
        return UncertaintyEvaluationResult(EuclideanEvaluation.name, result)


class JensenShannonEvaluation(UncertaintyEvaluationMetric):
    name = "jensen_shannon_distance"

    def _to_probability_distribution(self, ue: UncertaintyEstimate) -> np.ndarray:
        if ue.dimensions == 1:
            return ue.data / np.sum(ue.data)
        else:
            return np.array([d / np.sum(d) for d in ue.data])

    def _is_probability_distribution(self, data: np.ndarray) -> bool:
        return np.allclose(np.sum(data), 1.0)

    def evaluate(self, ue1: UncertaintyEstimate, ue2: UncertaintyEstimate) -> float:
        if ue1.dimensions != ue2.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        p1 = ue1.data
        p2 = ue2.data

        result = self._average_js_distance(p1, p2)
        return UncertaintyEvaluationResult(JensenShannonEvaluation.name, result)

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


class UncertaintyEvaluator:
    def __init__(self, eval_metric):
        self.eval_metric = eval_metric

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        import time
        id_time = list()
        ood_time = list()
        model.eval()
        id_ipt, id_opt = id_data
        ood_ipt, ood_opt = ood_data
        trials = 5
        with torch.no_grad():
            for t in range(trials):
                id_start = time.time()
                id_preds, id_ue_raw = model(id_ipt, return_ue=True)
                id_end = time.time()
                ood_start = time.time()
                ood_preds, ood_ue_raw = model(ood_ipt, return_ue=True)
                ood_end = time.time()

                id_time.append(id_end - id_start)
                ood_time.append(ood_end - ood_start)

        id_ue = UncertaintyEstimate(id_ue_raw)
        ood_ue = UncertaintyEstimate(ood_ue_raw)

        unc_results = self.eval_metric.evaluate(id_ue, ood_ue)
        loss_fn = model.val_loss

        return {
            "id_loss": loss_fn(id_preds, id_opt).item(),
            "ood_loss": loss_fn(ood_preds, ood_opt).item(),
            "avg_id_uncertainty": id_ue.mean().item(),
            "avg_ood_uncertainty": ood_ue.mean().item(),
            "uncertainty_dimensions": id_ue.dimensions,
            'id_time': id_time,
            'ood_time': ood_time,
            'id_ue': id_ue,
            'ood_ue': ood_ue,
            'uncertainty_evaluation': unc_results

        }


class ClassificationUncertaintyEvaluator(UncertaintyEvaluator):
    def __init__(self, eval_metric: UncertaintyEvaluationMetric):
        super().__init__(eval_metric)

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        import time
        id_time = list()
        ood_time = list()
        unc_evals = list()
        model.eval()
        id_ipt, id_opt = id_data
        ood_ipt, ood_opt = ood_data
        trials = 5
        with torch.no_grad():
            for t in range(trials):
                id_start = time.time()
                id_preds, id_ue = model(id_ipt, return_ue=True)
                id_end = time.time()
                ood_start = time.time()
                ood_preds, ood_ue = model(ood_ipt, return_ue=True)
                ood_end = time.time()

                id_time.append(id_end - id_start)
                ood_time.append(ood_end - ood_start)

        unc_metrics = self.eval_metric.evaluate(model, id_data, ood_data)
        unc_results = ClassificationEvaluationResult(self.eval_metric.get_metrics(), unc_metrics)
        id_ue = UncertaintyEstimate(id_ue)
        ood_ue = UncertaintyEstimate(ood_ue)
        loss_fn = model.val_loss

        return {
            "id_loss": loss_fn(id_preds, id_opt).item(),
            "ood_loss": loss_fn(ood_preds, ood_opt).item(),
            "avg_id_uncertainty": id_ue.mean().item(),
            "avg_ood_uncertainty": ood_ue.mean().item(),
            "uncertainty_dimensions": id_ue.dimensions,
            'id_time': id_time,
            'ood_time': ood_time,
            'id_ue': id_ue,
            'ood_ue': ood_ue,
            'uncertainty_evaluation': unc_results
        }


def get_uncertainty_evaluator(distance_metric: str | dict) -> UncertaintyEvaluator:
    distance_metrics = {
        WassersteinEvaluation.name: WassersteinEvaluation,
        EuclideanEvaluation.name: EuclideanEvaluation,
        JensenShannonEvaluation.name: JensenShannonEvaluation
    }

    if isinstance(distance_metric, str):
        metric = distance_metrics.get(distance_metric.lower())
    else:
        name, arg = distance_metric['name'], distance_metric['threshold']
        if name == 'percentile_classification':
            return ClassificationUncertaintyEvaluator(PercentileBasedIdOodClassifier(arg))
    if not metric:
        raise ValueError("Invalid distance metric type")

    return UncertaintyEvaluator(metric())
