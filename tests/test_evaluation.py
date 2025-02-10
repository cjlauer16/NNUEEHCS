import pytest
import torch
import numpy as np
from nnueehcs.evaluation import (WassersteinEvaluation, EuclideanEvaluation,
                                JensenShannonEvaluation, TNRatTPX,
                                UncertaintyEstimate, MetricEvaluator)
from nnueehcs.classification import (PercentileBasedIdOodClassifier, 
                                   ReversedPercentileBasedIdOodClassifier)


class DummyModel:
    """Mock model for testing evaluators"""
    def __init__(self, id_scores, ood_scores):
        self.id_scores = id_scores
        self.ood_scores = ood_scores
        self.device = 'cpu'
    
    def __call__(self, x, return_ue=False):
        if torch.equal(x, self.id_data[0]):
            return None, self.id_scores
        else:
            return None, self.ood_scores
    
    def eval(self):
        pass
    
    def to(self, device):
        return self


@pytest.fixture
def dummy_data():
    """Create dummy ID and OOD data for testing"""
    id_data = (torch.randn(100, 10), torch.randn(100, 1))
    ood_data = (torch.randn(100, 10), torch.randn(100, 1))
    return id_data, ood_data


@pytest.fixture
def dummy_model(dummy_data):
    """Create dummy model that returns predetermined uncertainty scores"""
    id_data, ood_data = dummy_data
    id_scores = torch.rand(100)
    ood_scores = torch.rand(100)
    model = DummyModel(id_scores, ood_scores)
    model.id_data = id_data
    model.ood_data = ood_data
    return model


def test_wasserstein_evaluation_returns_dict(dummy_model, dummy_data):
    """Test that WassersteinEvaluation returns a dictionary with expected keys"""
    evaluator = WassersteinEvaluation()
    result = evaluator.evaluate(dummy_model, dummy_data[0], dummy_data[1])
    
    assert isinstance(result, dict)
    assert "wasserstein_distance" in result
    assert isinstance(result["wasserstein_distance"], float)


def test_euclidean_evaluation_returns_dict(dummy_model, dummy_data):
    """Test that EuclideanEvaluation returns a dictionary with expected keys"""
    evaluator = EuclideanEvaluation()
    result = evaluator.evaluate(dummy_model, dummy_data[0], dummy_data[1])
    
    assert isinstance(result, dict)
    assert "euclidean_distance" in result
    assert isinstance(result["euclidean_distance"], float)


def test_jensen_shannon_evaluation_returns_dict(dummy_model, dummy_data):
    """Test that JensenShannonEvaluation returns a dictionary with expected keys"""
    evaluator = JensenShannonEvaluation()
    result = evaluator.evaluate(dummy_model, dummy_data[0], dummy_data[1])
    
    assert isinstance(result, dict)
    assert "jensen_shannon_distance" in result
    assert isinstance(result["jensen_shannon_distance"], float)


def test_tnr_at_tpr_evaluation_returns_dict(dummy_model, dummy_data):
    """Test that TNRatTPX returns a dictionary with expected keys"""
    evaluator = TNRatTPX(target_tpr=0.95)
    result = evaluator.evaluate(dummy_model, dummy_data[0], dummy_data[1])
    
    assert isinstance(result, dict)
    assert "tnr_at_tpr95" in result
    assert isinstance(result["tnr_at_tpr95"], float)


def test_percentile_classifier_returns_dict(dummy_model, dummy_data):
    """Test that PercentileBasedIdOodClassifier returns a dictionary with expected keys"""
    evaluator = PercentileBasedIdOodClassifier(percentile=0.95)
    result = evaluator.evaluate(dummy_model, dummy_data[0], dummy_data[1])
    
    assert isinstance(result, dict)
    assert "sensitivity" in result
    assert "specificity" in result
    assert "fpr" in result
    assert "fnr" in result


def test_uncertainty_estimate_creation():
    """Test that UncertaintyEstimate properly handles different input types"""
    # Test with numpy array
    np_data = np.random.rand(10, 10)
    ue = UncertaintyEstimate(np_data)
    assert isinstance(ue.data, np.ndarray)
    
    # Test with torch tensor
    torch_data = torch.rand(10, 10)
    ue = UncertaintyEstimate(torch_data)
    assert isinstance(ue.data, np.ndarray)
    
    # Test with tuple of arrays
    tuple_data = (np.random.rand(10, 10), np.random.rand(10, 10))
    ue = UncertaintyEstimate(tuple_data)
    assert isinstance(ue.data, tuple)
    assert all(isinstance(d, np.ndarray) for d in ue.data)


def test_evaluator_objectives_and_metrics():
    """Test that all evaluators properly define their objectives and metrics"""
    evaluators = [
        WassersteinEvaluation(),
        EuclideanEvaluation(),
        JensenShannonEvaluation(),
        TNRatTPX(target_tpr=0.95),
        PercentileBasedIdOodClassifier(percentile=0.95)
    ]
    
    for evaluator in evaluators:
        objectives = evaluator.get_objectives()
        metrics = evaluator.get_metrics()
        
        assert isinstance(objectives, list)
        assert isinstance(metrics, list)
        assert all(isinstance(obj, dict) for obj in objectives)
        assert all(isinstance(metric, str) for metric in metrics)
        
        # Check that each objective has required keys
        for obj in objectives:
            assert "name" in obj
            assert "type" in obj
            assert obj["type"] in ["maximize", "minimize"]


def test_uncertainty_estimate_edge_cases():
    """Test UncertaintyEstimate with edge cases and potential problem inputs"""
    # Empty arrays
    empty_array = np.array([])
    with pytest.raises(ValueError):
        UncertaintyEstimate(empty_array)
    
    # NaN values
    nan_array = np.array([np.nan, 1.0, 2.0])
    ue = UncertaintyEstimate(nan_array)
    assert np.isnan(ue.mean())
    
    # Infinity values
    inf_array = np.array([np.inf, 1.0, 2.0])
    ue = UncertaintyEstimate(inf_array)
    assert np.isinf(ue.mean())
    
    # Mixed types in tuple
    mixed_tuple = (torch.randn(10), np.random.rand(10))
    ue = UncertaintyEstimate(mixed_tuple)
    assert all(isinstance(d, np.ndarray) for d in ue.data)


def test_uncertainty_estimate_dimensions():
    """Test dimension handling in UncertaintyEstimate"""
    # Single dimension
    single_dim = np.random.rand(10)
    ue = UncertaintyEstimate(single_dim)
    assert ue.dimensions == 1
    
    # Multiple dimensions
    multi_dim = (np.random.rand(10), np.random.rand(10))
    ue = UncertaintyEstimate(multi_dim)
    assert ue.dimensions == 2
    
    # Mismatched dimensions in tuple
    mismatched = (np.random.rand(10), np.random.rand(5))
    with pytest.raises(ValueError):
        UncertaintyEstimate(mismatched)


def test_wasserstein_evaluation_edge_cases(dummy_model):
    """Test WassersteinEvaluation with edge cases"""
    evaluator = WassersteinEvaluation()
    
    # Test with identical distributions
    identical_data = torch.ones(100)
    result = evaluator._evaluate_uncertainties(
        UncertaintyEstimate(identical_data),
        UncertaintyEstimate(identical_data)
    )
    assert result["wasserstein_distance"] == 0.0
    
    # Test with completely different distributions
    dist1 = torch.zeros(100)
    dist2 = torch.ones(100)
    result = evaluator._evaluate_uncertainties(
        UncertaintyEstimate(dist1),
        UncertaintyEstimate(dist2)
    )
    assert result["wasserstein_distance"] > 0.0


def test_tnr_at_tpr_edge_cases():
    """Test TNRatTPX with edge cases"""
    # Invalid target_tpr
    with pytest.raises(ValueError):
        TNRatTPX(target_tpr=1.5)
    with pytest.raises(ValueError):
        TNRatTPX(target_tpr=-0.1)
        
    evaluator = TNRatTPX(target_tpr=0.95)
    
    # Perfect separation
    id_scores = torch.zeros(100)
    ood_scores = torch.ones(100)
    result = evaluator._evaluate_scores(id_scores, ood_scores)
    assert result["tnr_at_tpr95"] == 1.0
    
    # Complete overlap
    id_scores = torch.ones(100)
    ood_scores = torch.ones(100)
    result = evaluator._evaluate_scores(id_scores, ood_scores)
    assert result["tnr_at_tpr95"] == 0.0


def test_percentile_classifier_edge_cases():
    """Test PercentileBasedIdOodClassifier with edge cases"""
    # Invalid percentile
    with pytest.raises(ValueError):
        PercentileBasedIdOodClassifier(percentile=1.5)
    
    classifier = PercentileBasedIdOodClassifier(percentile=0.95)
    
    # Perfect separation
    id_scores = torch.zeros(100)
    ood_scores = torch.ones(100)
    result = classifier._evaluate_scores(id_scores, ood_scores)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.95  # Due to percentile threshold
    
    # Complete overlap
    id_scores = torch.ones(100)
    ood_scores = torch.ones(100)
    result = classifier._evaluate_scores(id_scores, ood_scores)
    assert result["sensitivity"] == 0.05  # Due to percentile threshold
    assert result["specificity"] == 0.95


def test_reversed_percentile_classifier():
    """Test ReversedPercentileBasedIdOodClassifier behavior"""
    classifier = ReversedPercentileBasedIdOodClassifier(percentile=0.95)
    
    # Test that reversed classifier gives opposite results
    id_scores = torch.zeros(100)
    ood_scores = torch.ones(100)
    
    normal_classifier = PercentileBasedIdOodClassifier(percentile=0.95)
    normal_result = normal_classifier._evaluate_scores(id_scores, ood_scores)
    reversed_result = classifier._evaluate_scores(id_scores, ood_scores)
    
    assert abs(normal_result["sensitivity"] - (1 - reversed_result["sensitivity"])) < 1e-6
    assert abs(normal_result["specificity"] - reversed_result["specificity"]) < 1e-6


def test_metric_evaluator_combination():
    """Test MetricEvaluator with multiple metrics"""
    metrics = [
        WassersteinEvaluation(),
        EuclideanEvaluation(),
        TNRatTPX(target_tpr=0.95)
    ]
    
    evaluator = MetricEvaluator(metrics)
    
    # Check that objectives are combined correctly
    objectives = evaluator.get_training_objectives()
    assert len(objectives) == 3
    assert all("type" in obj for obj in objectives)
    
    # Check that metrics are combined correctly
    all_metrics = evaluator.get_all_metrics()
    assert len(all_metrics) == 3
    assert "wasserstein_distance" in all_metrics
    assert "euclidean_distance" in all_metrics
    assert "tnr_at_tpr95" in all_metrics


def test_evaluator_numerical_stability():
    """Test numerical stability of evaluators with extreme values"""
    # Create data with extreme values
    extreme_data = torch.tensor([1e-10, 1e10, 1e-10, 1e10])
    normal_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    evaluators = [
        WassersteinEvaluation(),
        EuclideanEvaluation(),
        JensenShannonEvaluation()
    ]
    
    for evaluator in evaluators:
        result = evaluator._evaluate_uncertainties(
            UncertaintyEstimate(extreme_data),
            UncertaintyEstimate(normal_data)
        )
        # Results should be finite
        assert all(np.isfinite(v) for v in result.values())
