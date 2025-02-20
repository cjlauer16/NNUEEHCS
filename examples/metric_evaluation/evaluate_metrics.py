from nnueehcs.evaluation import get_evaluator
from nnueehcs.utility import ResultsComposite, ResultsInstance
from nnueehcs.data_utils import get_dataset, prepare_dataset_for_use
from pathlib import Path
import yaml
import sys
import click
import pandas as pd
import torch

def get_evaluators(metrics: list[dict]):
    evaluators = []
    for metric in metrics:
        evaluators.append(get_evaluator(metric))
    return evaluators

def RMSE(output, target):
    return torch.sqrt(torch.mean((output - target) ** 2))

def find_best_training_run(results_instance: ResultsInstance, train_eval_metric) -> tuple[float, pd.Series]:
    """
    Find the training run that maximized the training metric.
    
    Args:
        results_instance: ResultsInstance object containing the results
        train_eval_metric: The metric used during training
    
    Returns:
        tuple of (max metric value, pandas Series containing the best run info)
    """
    res = pd.read_csv(results_instance.get_trial_results_file())
    train_metric_name = train_eval_metric.get_metrics()[0]
    max_train_metric_value = res[train_metric_name].max()
    max_train_metric_instance = res[res[train_metric_name] == max_train_metric_value].iloc[0]
    return max_train_metric_value, max_train_metric_instance

def get_latest_finished_trial(composite, benchmark: str, dataset_name: str, method: str) -> ResultsInstance:
    """
    Find the most recent trial that has completed (produced results).
    
    Args:
        composite: ResultsComposite object containing all results
        benchmark: Name of the benchmark
        dataset_name: Name of the dataset
        method: Name of the method
    
    Returns:
        ResultsInstance for the latest finished trial
    """
    num_trials = composite.get_num_trials(benchmark, dataset_name, method)
    results_instance = composite.get_results_instance(benchmark, dataset_name, method, f'bo_trial_{num_trials - 1}')
    while not results_instance.is_finished():
        num_trials -= 1
        results_instance = composite.get_results_instance(benchmark, dataset_name, method, f'bo_trial_{num_trials - 1}')
    return results_instance

def get_benchmark_dataset_pairs(composite: ResultsComposite, benchmark: str = None, dataset: str = None) -> list[tuple[str, str]]:
    """Get all valid benchmark-dataset pairs to evaluate.
    
    Args:
        composite: ResultsComposite object containing results
        benchmark: Optional benchmark name to filter by
        dataset: Optional dataset name to filter by
        
    Returns:
        List of (benchmark, dataset) tuples to evaluate
    """
    pairs = []
    benchmarks = [benchmark] if benchmark else list(composite.get_benchmark_names())
    
    for bench in benchmarks:
        datasets = [dataset] if dataset else list(composite.get_dataset_names(bench))
        for ds in datasets:
            if list(composite.get_method_names(bench, ds)):
                pairs.append((bench, ds))
            else:
                print(f"Warning: Skipping {bench}/{ds} - no methods found")
                
    return pairs

def prepare_datasets(dataset_cfg, dataset_name, training_cfg):
    """Prepare both in-distribution and out-of-distribution datasets.
    
    Returns:
        tuple: (dataset_id, dataset_ood) - both datasets moved to appropriate device
    """
    dataset_id = get_dataset(dataset_cfg, dataset_name)
    dataset_ood = get_dataset(dataset_cfg, dataset_name, is_ood=True)
    
    dataset_ood = prepare_dataset_for_use(dataset_ood, training_cfg, scaling_dset=dataset_id)
    dataset_id = prepare_dataset_for_use(dataset_id, training_cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dataset_id.to(device), dataset_ood.to(device)

def load_best_model(composite, benchmark, dataset, method, train_eval_metric):
    """Load the best performing model from all trials.
    
    Returns:
        tuple: (model, best_run_trial, device)
    """
    results_instance = get_latest_finished_trial(composite, benchmark, dataset, method)
    max_value, best_run = find_best_training_run(results_instance, train_eval_metric)
    best_run_trial = Path(best_run['log_path']).stem
    
    # Get model
    best_run_instance = composite.get_results_instance(benchmark, dataset, method, best_run_trial)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(best_run_instance.get_model_file(), map_location=device)
    model.eval()
    
    return model, best_run_trial, device

def evaluate_model_metrics(model, dataset_id, dataset_ood, evaluators):
    """Evaluate all metrics for a given model and datasets.
    
    Returns:
        list: List of evaluation results as [benchmark, dataset, method, trial, metric, objective, value]
    """
    results = []
    with torch.no_grad():
        for metric in evaluators.metrics:
            print(f"Evaluating with {metric.get_name()}")
            result = metric.evaluate(model, (dataset_id.input, dataset_id.output),
                                   (dataset_ood.input, dataset_ood.output))
            
            for objective_name, objective_value in result.items():
                results.append([metric.get_name(), objective_name, objective_value])
    return results

def find_all_training_runs(results_instance: ResultsInstance) -> list[pd.Series]:
    """
    Get all completed training runs.
    
    Args:
        results_instance: ResultsInstance object containing the results
    
    Returns:
        list of pandas Series containing all run info
    """
    res = pd.read_csv(results_instance.get_trial_results_file())
    return [row for _, row in res.iterrows()]

def process_benchmark_dataset(composite, config, benchmark, dataset, evaluators, evaluate_all: bool = False):
    """Process a single benchmark-dataset pair and return evaluation results."""
    print(f"\nProcessing benchmark {benchmark}, dataset {dataset}")
    
    # Get configurations
    dataset_cfg = config['benchmarks'][benchmark]['datasets']
    training_cfg = config['training']
    train_eval_metric = get_evaluator(config['bo_config']['evaluation_metric']).metrics[0]
    print(f"Using training evaluation metric: {train_eval_metric.get_name()}")
    
    dataset_id, dataset_ood = prepare_datasets(dataset_cfg, dataset, training_cfg)
    
    results = []
    methods = list(composite.get_method_names(benchmark, dataset))
    for method in methods:
        print(f"\nEvaluating method: {method}")
        
        results_instance = get_latest_finished_trial(composite, benchmark, dataset, method)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if evaluate_all:
            runs = find_all_training_runs(results_instance)
        else:
            _, best_run = find_best_training_run(results_instance, train_eval_metric)
            runs = [best_run]
            
        for run in runs:
            trial = Path(run['log_path']).stem
            print(f"Evaluating trial: {trial}")
            
            # Get model for this trial
            trial_instance = composite.get_results_instance(benchmark, dataset, method, trial)
            model = torch.load(trial_instance.get_model_file(), map_location=device)
            model.eval()
            
            metric_results = evaluate_model_metrics(model, dataset_id, dataset_ood, evaluators)
            
            for metric_name, objective_name, value in metric_results:
                results.append([benchmark, dataset, method, trial, metric_name, objective_name, value])
    
    return results

@click.command("Post-hoc application of metrics to results")
@click.option("--results_dir", type=click.Path(exists=True), help="The directory containing the results")
@click.option("--config_file", type=click.Path(exists=True), help="The config file containing the metrics to evaluate")
@click.option("--benchmark", type=str, help="The benchmark to evaluate (optional)", required=False)
@click.option("--dataset", type=str, help="The dataset to evaluate (optional)", required=False)
@click.option("--output", type=str, help="The output file name", default="evaluated_metrics.csv")
@click.option("--evaluate_all", is_flag=True, help="Evaluate all models instead of just the best one")
def evaluate_metrics(results_dir: str, config_file: str, benchmark: str, dataset: str, output: str, evaluate_all: bool):
    composite = ResultsComposite(results_dir)
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    
    # Get configuration
    eval_config = config["evaluation"]
    metrics = eval_config["metrics"]
    training_cfg = config['training']
    evaluators = get_evaluator(metrics)
    
    # Get all benchmark-dataset pairs to evaluate
    pairs_to_evaluate = get_benchmark_dataset_pairs(composite, benchmark, dataset)
    if not pairs_to_evaluate:
        raise ValueError("No valid benchmark-dataset pairs found to evaluate")
    
    # Setup results collection
    columns = ['benchmark', 'dataset', 'method', 'trial', 'metric', 'objective', 'value']
    rows = []
    
    # Process each benchmark-dataset pair
    for current_benchmark, current_dataset in pairs_to_evaluate:
        results = process_benchmark_dataset(composite, config, current_benchmark, current_dataset, evaluators, evaluate_all)
        rows.extend(results)
    
    # Save results
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f"{output}", index=False)
    print(f"\nResults saved to {output}")

if __name__ == "__main__":
    evaluate_metrics()