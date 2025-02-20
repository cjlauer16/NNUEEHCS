from pathlib import Path
import os
from typing import Callable
import re
import pandas as pd

DefaultFileMap = {
    "optimization_step": "ax_client_optimization_step.json",
    "model": "model.pth",
    "trial_results": "trial_results.csv",
    "training_metrics": "metrics.csv",
}

DefaultTrialDirRegex = r"^bo_trial_(\d+)$"

class ResultsInstance:
    def __init__(self, results_dir: str, filemap: dict = DefaultFileMap, trial_dir_regex: str = DefaultTrialDirRegex):
        self.results_dir = results_dir
        self.filemap = filemap
        self.files = self._get_files(results_dir)
        self.trial_dir_regex = trial_dir_regex

    def load(self):
        pass

    def _get_files(self, results_dir: str):
        files = {}
        for key, value in self.filemap.items():
            files[key] = os.path.join(results_dir, value)
        return files

    def get_model_file(self):
        return self.files["model"]

    def get_trial_results_file(self):
        return self.files["trial_results"]

    def get_training_metrics_file(self):
        return self.files["training_metrics"]

    def get_optimization_step_file(self):
        return self.files["optimization_step"]

    def get_benchmark_name(self):
        return Path(self.results_dir).parent.parent.parent.stem

    def get_dataset_name(self):
        return Path(self.results_dir).parent.parent.stem

    def get_method_name(self):
        return Path(self.results_dir).parent.stem

    def get_trial_name(self):
        return Path(self.results_dir).stem

    def get_trial_number(self):
        # apply trial_dir_regex to the results_dir
        name = self.get_trial_name()
        match = re.match(self.trial_dir_regex, name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Trial name {name} does not match regex {self.trial_dir_regex}")

    def get_metric(self, metric_name: str):
        return self.get_results()[metric_name]

    def get_results(self):
        full_path = Path(self.results_dir) / self.files["trial_results"]
        return pd.read_csv(full_path)

    def is_finished(self):
        return os.path.exists(self.get_trial_results_file())

    def __str__(self):
        return f"{self.results_dir}"

class ResultsComposite:
    """
    This class represents a composite of results.
    It is used to represent a collection of results across
    multiple benchmarks, datasets, UQ methods, and trials.
    You can compute metrics on each results instance, filter
    by metrics, and so on.

    Assumes the structure of the results is as follows:
    results/
      benchmark_1/
        dataset_1/
          method_1/
            bo_trial_1/
              ...
    """
    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    def get_benchmark_names(self):
        yield from [x.stem for x in Path(self.results_dir).glob(f"*")]
    
    def get_dataset_names(self, benchmark_name: str = None):
        """
        Returns all datasets in the results directory across all benchmarks.
        """
        if benchmark_name is None:
            benchmark_names = self.get_benchmark_names()
        else:
            benchmark_names = [benchmark_name]
        for benchmark_name in benchmark_names:
            yield from sorted(set([x.stem for x in Path(os.path.join(self.results_dir, benchmark_name)).glob(f"*")]))
    
    def get_method_names(self, benchmark_name: str = None, dataset_name: str = None):
        """
        Returns all methods in the results directory across all benchmarks and datasets.
        """
        if benchmark_name is None:
            benchmark_names = self.get_benchmark_names()
        else:
            benchmark_names = [benchmark_name]

        if dataset_name is None:
            dataset_names = self.get_dataset_names(benchmark_name)
        else:
            dataset_names = [dataset_name]

        for benchmark_name in benchmark_names:
            for dataset_name in dataset_names:
                yield from sorted(set([x.stem for x in Path(os.path.join(self.results_dir, benchmark_name, dataset_name)).glob(f"*")]))

    def get_trial_names(self, benchmark_name: str = None, dataset_name: str = None, method_name: str = None):
        """
        Returns all trials in the results directory across all benchmarks, datasets, and methods.
        """
        if benchmark_name is None:
            benchmark_names = self.get_benchmark_names()
        else:
            benchmark_names = [benchmark_name]
        if dataset_name is None:
            dataset_names = self.get_dataset_names(benchmark_name)
        else:
            dataset_names = [dataset_name]
        if method_name is None:
            method_names = self.get_method_names(benchmark_name, dataset_name)
        else:
            method_names = [method_name]

        for benchmark_name in benchmark_names:
            for dataset_name in dataset_names:
                for method_name in method_names:
                    yield from sorted(set([x.stem for x in Path(os.path.join(self.results_dir, benchmark_name, dataset_name, method_name)).glob(f"*")]))


    def get_results(self):
        """
        Returns all results in the results directory across all benchmarks, datasets, methods, and trials.
        """
        for benchmark_name in self.get_benchmark_names():
            for dataset_name in self.get_dataset_names(benchmark_name):
                for method_name in self.get_method_names(benchmark_name, dataset_name):
                    for trial_name in self.get_trial_names(benchmark_name, dataset_name, method_name):
                        yield self.get_results_instance(benchmark_name, dataset_name, method_name, trial_name)
    

    def get_results_instance(self, benchmark_name: str, dataset_name: str, method_name: str, trial_name: str):
        return ResultsInstance(os.path.join(self.results_dir, benchmark_name, dataset_name, method_name, trial_name))
    
    def get_num_trials(self, benchmark_name: str, dataset_name: str, method_name: str):
        return len(list(self.get_trial_names(benchmark_name, dataset_name, method_name)))
    
    def get_num_methods(self, benchmark_name: str, dataset_name: str):
        return len(list(self.get_method_names(benchmark_name, dataset_name)))
    
    def get_num_datasets(self, benchmark_name: str):
        return len(list(self.get_dataset_names(benchmark_name)))
    
    def get_num_benchmarks(self):
        return len(list(self.get_benchmark_names()))
        
    def apply_functor(self, functor: Callable):
        """
        Apply a functor to each results instance in the composite.
        The functor should take a ResultsInstance and return a dictionary of results.
        The results are returned as a dictionary of dictionaries of dictionaries of dictionaries.
        """
        results = {}
        for benchmark_name in self.get_benchmark_names():
            for dataset_name in self.get_dataset_names(benchmark_name):
                for method_name in self.get_method_names(benchmark_name, dataset_name):
                    for trial_name in self.get_trial_names(benchmark_name, dataset_name, method_name):
                        results[benchmark_name, dataset_name, method_name, trial_name] = functor(self.get_results_instance(benchmark_name, dataset_name, method_name, trial_name))
        return results

    def filter_by_metric(self, metric_name: str):
        """
        Filter the results by a metric.
        The metric should be a string that is a key in the results dictionary.
        """
        return self.apply_functor(lambda x: x.get_metric(metric_name))

    def filter(self, functor: Callable):
        """
        Filter the results by applying a functor to each results instance.
        The functor should take a list of results and apply a filter to it.
        """
        res = self.apply_functor(lambda x: (functor(x), x))
        true_dict = {}
        for key, value in res.items():
            if value[0]:
                true_dict[key] = value[1]
        return true_dict

    def filter_by_benchmark(self, benchmark_name: str):
        """
        Filter the results by benchmark.
        """
        return self.filter(lambda x: x.get_benchmark_name() == benchmark_name)
    
    def filter_by_dataset(self, dataset_name: str):
        """
        Filter the results by dataset.
        """
        return self.filter(lambda x: x.get_dataset_name() == dataset_name)
    
    def filter_by_method(self, method_name: str):
        """
        Filter the results by method.
        """
        return self.filter(lambda x: x.get_method_name() == method_name)
    
    def filter_by_trial(self, trial_name: str):
        """
        Filter the results by trial.
        """
        return self.filter(lambda x: x.get_trial_name() == trial_name)
