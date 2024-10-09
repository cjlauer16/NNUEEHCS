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

        return {
            "id_preds": id_preds,
            "ood_preds": ood_preds,
            "id_scores": id_scores,
            "ood_scores": ood_scores
        }


class PercentileBasedIdOodClassifier(_IdOodClassifier):
    def __init__(self, percentile: float):
        super().__init__()
        self.percentile = percentile

    def _fpr(self, false_positives: int, true_negatives: int) -> float:
        return false_positives / (false_positives + true_negatives)

    def _fnr(self, false_negatives: int, true_positives: int) -> float:
        return false_negatives / (false_negatives + true_positives)

    def _sensitivity(self, true_positives: int, false_negatives: int) -> float:
        return true_positives / (true_positives + false_negatives)

    def _specificity(self, true_negatives: int, false_positives: int) -> float:
        return true_negatives / (true_negatives + false_positives)

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        id_ipt, id_opt = id_data
        ood_ipt, ood_opt = ood_data
        results = super().evaluate(model, id_data, ood_data)

        id_scores = results['id_scores']
        ood_scores = results['ood_scores']

        id_percentile = torch.quantile(id_scores, self.percentile)
        id_above = id_scores[id_scores > id_percentile]
        id_below = id_scores[id_scores <= id_percentile]

        ood_above = ood_scores[ood_scores > id_percentile]
        ood_below = ood_scores[ood_scores <= id_percentile]

        false_positives = len(id_above)
        false_negatives = len(ood_below)

        true_positives = len(ood_above)
        true_negatives = len(id_below)

        fpr = self._fpr(false_positives, true_negatives)
        fnr = self._fnr(false_negatives, true_positives)
        sensitivity = self._sensitivity(true_positives, false_negatives)
        specificity = self._specificity(true_negatives, false_positives)

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            'fpr': fpr,
            'fnr': fnr
        }

    @classmethod
    def get_objectives(cls):
        # Specificity is always going to be instance.percentile
        # So we don't need to include it in the objectives
        return [
            {'name': 'sensitivity',
             'type': 'maximize'
             }
            # {
            #     'name': 'specificity',
            #     'type': 'maximize'
            # }
        ]

    @classmethod
    def get_metrics(cls):
        return [
            'sensitivity',
            # 'specificity'
        ]