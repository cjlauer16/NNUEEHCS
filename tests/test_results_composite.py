import pytest
import os
import tempfile
import pandas as pd
from nnueehcs.utility import ResultsComposite, ResultsInstance

# Fixture of a results directory to test the composite
# Needs to take the structure like this:
# results/
#   benchmark_1/
#     dataset_1/
#       method_1/
#         bo_trial_1/
#           ...
#         bo_trial_2/
#           ...
#         ...
#       ...
#     ...
#   ...
# With two benchmarks, two datasets, two methods, and two trials.
@pytest.fixture
def results_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the results directory
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir)

        # Create the benchmarks
        benchmark_1 = os.path.join(results_dir, "benchmark_1")
        os.makedirs(benchmark_1)

        # Create the datasets
        dataset_1 = os.path.join(benchmark_1, "dataset_1")
        os.makedirs(dataset_1)

        # Create the methods
        method_1 = os.path.join(dataset_1, "method_1")
        os.makedirs(method_1)

        method_2 = os.path.join(dataset_1, "method_2")
        os.makedirs(method_2)

        # Create the trials
        trial_1 = os.path.join(method_1, "bo_trial_1")
        os.makedirs(trial_1)

        trial_2 = os.path.join(method_1, "bo_trial_2")
        os.makedirs(trial_2)

        trial_3 = os.path.join(method_2, "bo_trial_1")
        os.makedirs(trial_3)

        trial_4 = os.path.join(method_2, "bo_trial_2")
        os.makedirs(trial_4)

        # Create the files
        file_1 = os.path.join(trial_1, "trial_results.csv")
        with open(file_1, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("1,2,3\n")

        file_2 = os.path.join(trial_2, "trial_results.csv")
        with open(file_2, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("4,5,6\n")

        file_3 = os.path.join(trial_3, "trial_results.csv")
        with open(file_3, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("7,8,9\n")

        file_4 = os.path.join(trial_4, "trial_results.csv")
        with open(file_4, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("10,11,12\n")


        benchmark_2 = os.path.join(results_dir, "benchmark_2")
        os.makedirs(benchmark_2)

        dataset_2 = os.path.join(benchmark_2, "dataset_1")
        os.makedirs(dataset_2)

        method_3 = os.path.join(dataset_2, "method_1")
        os.makedirs(method_3)

        method_4 = os.path.join(dataset_2, "method_2")
        os.makedirs(method_4)

        trial_5 = os.path.join(method_3, "bo_trial_1")
        os.makedirs(trial_5)

        trial_6 = os.path.join(method_3, "bo_trial_2")
        os.makedirs(trial_6)

        trial_7 = os.path.join(method_4, "bo_trial_1")
        os.makedirs(trial_7)

        trial_8 = os.path.join(method_4, "bo_trial_2")
        os.makedirs(trial_8)

        file_5 = os.path.join(trial_5, "trial_results.csv")
        with open(file_5, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("7,8,9\n")

        file_6 = os.path.join(trial_6, "trial_results.csv")
        with open(file_6, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("10,11,12\n")

        file_7 = os.path.join(trial_7, "trial_results.csv")
        with open(file_7, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("13,14,15\n")

        file_8 = os.path.join(trial_8, "trial_results.csv")
        with open(file_8, "w") as f:
            f.write("metric_1,metric_2,metric_3\n")
            f.write("16,17,18\n")

        yield results_dir


def test_results_composite(results_dir):
    composite = ResultsComposite(results_dir)
    assert set(composite.get_benchmark_names()) == set(["benchmark_1", "benchmark_2"])
    assert set(composite.get_dataset_names()) == set(["dataset_1"])
    assert set(composite.get_method_names()) == set(["method_1", "method_2"])
    assert set(composite.get_trial_names()) == set(["bo_trial_1", "bo_trial_2"])

    

def test_results_instance(results_dir):
    full_path = os.path.join(results_dir, "benchmark_1", "dataset_1", "method_1", "bo_trial_1")
    instance = ResultsInstance(full_path)
    assert instance.get_benchmark_name() == "benchmark_1"
    assert instance.get_dataset_name() == "dataset_1"
    assert instance.get_method_name() == "method_1"
    assert instance.get_trial_name() == "bo_trial_1"
    
    assert instance.get_trial_number() == 1
    assert instance.get_trial_results_file() == os.path.join(full_path, "trial_results.csv")
    assert (instance.get_metric("metric_1") == pd.DataFrame({"metric_1": [1], "metric_2": [2], "metric_3": [3]})["metric_1"]).all()


def test_filter_by_metric(results_dir):
    composite = ResultsComposite(results_dir)
    filtered = composite.filter_by_metric("metric_1")
    assert len(filtered) == 8
    assert set(filtered.keys()) == set([("benchmark_1", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_1", "bo_trial_2"),
                                        ("benchmark_1", "dataset_1", "method_2", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_2", "bo_trial_2"),
                                        ("benchmark_2", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_2", "dataset_1", "method_1", "bo_trial_2"),
                                        ("benchmark_2", "dataset_1", "method_2", "bo_trial_1"), 
                                        ("benchmark_2", "dataset_1", "method_2", "bo_trial_2")])
    
    vals = list(filtered.values())
    assert (vals[0] == pd.DataFrame({"metric_1": [1]})["metric_1"]).all()
    assert (vals[1] == pd.DataFrame({"metric_1": [4]})["metric_1"]).all()
    assert (vals[2] == pd.DataFrame({"metric_1": [7]})["metric_1"]).all()
    assert (vals[3] == pd.DataFrame({"metric_1": [10]})["metric_1"]).all()
    assert (vals[4] == pd.DataFrame({"metric_1": [7]})["metric_1"]).all()
    assert (vals[5] == pd.DataFrame({"metric_1": [10]})["metric_1"]).all()
    assert (vals[6] == pd.DataFrame({"metric_1": [13]})["metric_1"]).all()
    assert (vals[7] == pd.DataFrame({"metric_1": [16]})["metric_1"]).all()

def test_filter_by_benchmark(results_dir):
    composite = ResultsComposite(results_dir)
    filtered = composite.filter_by_benchmark("benchmark_1")
    assert len(filtered) == 4
    assert set(filtered.keys()) == set([("benchmark_1", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_1", "bo_trial_2"),
                                        ("benchmark_1", "dataset_1", "method_2", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_2", "bo_trial_2")])
    
    vals = list(filtered.values())
    assert (vals[0].get_metric("metric_1") == pd.DataFrame({"metric_1": [1]})["metric_1"]).all()
    assert (vals[1].get_metric("metric_1") == pd.DataFrame({"metric_1": [4]})["metric_1"]).all()
    assert (vals[2].get_metric("metric_1") == pd.DataFrame({"metric_1": [7]})["metric_1"]).all()
    assert (vals[3].get_metric("metric_1") == pd.DataFrame({"metric_1": [10]})["metric_1"]).all()

def test_filter_by_dataset(results_dir):
    composite = ResultsComposite(results_dir)
    filtered = composite.filter_by_dataset("dataset_1")
    assert len(filtered) == 8
    assert set(filtered.keys()) == set([("benchmark_1", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_1", "bo_trial_2"),
                                        ("benchmark_1", "dataset_1", "method_2", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_2", "bo_trial_2"),
                                        ("benchmark_2", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_2", "dataset_1", "method_1", "bo_trial_2"),
                                        ("benchmark_2", "dataset_1", "method_2", "bo_trial_1"),
                                        ("benchmark_2", "dataset_1", "method_2", "bo_trial_2")])
    
    vals = list(filtered.values())
    assert (vals[0].get_metric("metric_1") == pd.DataFrame({"metric_1": [1]})["metric_1"]).all()
    assert (vals[1].get_metric("metric_1") == pd.DataFrame({"metric_1": [4]})["metric_1"]).all()
    assert (vals[2].get_metric("metric_1") == pd.DataFrame({"metric_1": [7]})["metric_1"]).all()
    assert (vals[3].get_metric("metric_1") == pd.DataFrame({"metric_1": [10]})["metric_1"]).all()

def test_filter_by_method(results_dir):
    composite = ResultsComposite(results_dir)
    filtered = composite.filter_by_method("method_1")
    assert len(filtered) == 4
    assert set(filtered.keys()) == set([("benchmark_1", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_1", "dataset_1", "method_1", "bo_trial_2"),
                                        ("benchmark_2", "dataset_1", "method_1", "bo_trial_1"),
                                        ("benchmark_2", "dataset_1", "method_1", "bo_trial_2")])
    
    vals = list(filtered.values())
    assert (vals[0].get_metric("metric_1") == pd.DataFrame({"metric_1": [1]})["metric_1"]).all()
    assert (vals[1].get_metric("metric_1") == pd.DataFrame({"metric_1": [4]})["metric_1"]).all()
    assert (vals[2].get_metric("metric_1") == pd.DataFrame({"metric_1": [7]})["metric_1"]).all()
    assert (vals[3].get_metric("metric_1") == pd.DataFrame({"metric_1": [10]})["metric_1"]).all()
    
