from nnueehcs.classification import PercentileBasedIdOodClassifier
import torch


def test_percentile_based_classifier():
    classifier = PercentileBasedIdOodClassifier(0.5)
    assert classifier.percentile == 0.5

    id_data = (torch.arange(1, 11, dtype=torch.float64), torch.randint(0, 2, (10,)))
    ood_data = (torch.arange(1, 11, dtype=torch.float64), torch.randint(0, 2, (10,)))

    class Model:
        def __call__(self, x, return_ue=False):
            return torch.randn(x.shape[0]), x

        def eval(self):
            pass

    assert classifier.percentile == 0.5
    model = Model()
    results = classifier.evaluate(model, id_data, ood_data)
    assert results['fpr'] == 0.5
    assert results['fnr'] == 0.5
    assert results['sensitivity'] == 0.5
    assert results['specificity'] == 0.5
