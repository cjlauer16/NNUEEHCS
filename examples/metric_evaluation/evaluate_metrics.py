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

@click.command("Post-hoc application of metrics to results")
@click.option("--results_dir", type=click.Path(exists=True), help="The directory containing the results")
@click.option("--config_file", type=click.Path(exists=True), help="The config file containing the metrics to evaluate")
@click.option("--benchmark", type=str, help="The benchmark to evaluate")
@click.option("--dataset", type=str, help="The dataset to evaluate")
@click.option("--output", type=str, help="The output file name")
def evaluate_metrics(results_dir: str, config_file: str, benchmark: str, dataset: str, output: str):
    dataset_name = dataset
    composite = ResultsComposite(results_dir)
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    eval_config = config["evaluation"]
    metrics = eval_config["metrics"]
    dataset_cfg = config['benchmarks'][benchmark]['datasets']
    training_cfg = config['training']
    evaluators = get_evaluator(metrics)
    dataset_id = get_dataset(dataset_cfg, dataset_name)
    dataset_ood = get_dataset(dataset_cfg, dataset_name, is_ood=True)


    train_eval_metric = config['bo_config']['evaluation_metric']
    train_eval_metric = get_evaluator(train_eval_metric).metrics[0]
    print(f"Eval metric name: {train_eval_metric.get_name()}")
    # we have to scale ood relative to ID
    # thus, we must scale OOD first because this method
    # modifies in place
    dataset_ood = prepare_dataset_for_use(dataset_ood, training_cfg, scaling_dset=dataset_id)
    dataset_id = prepare_dataset_for_use(dataset_id, training_cfg)
    print(evaluators.metrics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = dataset_id.to(device)
    dataset_ood = dataset_ood.to(device)

    ipt_combined = torch.cat((dataset_id.input, dataset_ood.input))
    opt_combined = torch.cat((dataset_id.output, dataset_ood.output))

    print(f"Benchmark is {benchmark} and dataset is {dataset_name}")
    columns = ['benchmark', 'dataset', 'method', 'trial', 'metric','objective', 'value']

    rows = []
    this_ds_and_benchmark = lambda x: x.get_dataset_name() == dataset_name and x.get_benchmark_name() == benchmark
    methods = composite.get_method_names(benchmark, dataset_name)
    for method in methods:
        results_instance = get_latest_finished_trial(composite, benchmark, dataset_name, method)
        
        max_value, best_run = find_best_training_run(results_instance, train_eval_metric)
        best_run_trial = Path(best_run['log_path']).stem

        # Get the results instance for the specific training run
        best_run_instance = composite.get_results_instance(benchmark, dataset_name, method, best_run_trial)
        model_file = best_run_instance.get_model_file()
        model = torch.load(model_file, map_location=device)
        model.eval()

        with torch.no_grad():
            for metric in evaluators.metrics:
                print(f"Evaluating benchmark {benchmark}, dataset {dataset_name}, method {method}, trial {best_run_trial} with {metric.get_name()}")
                result = metric.evaluate(model, (dataset_id.input, dataset_id.output), 
                                      (dataset_ood.input, dataset_ood.output))
                print(f"{metric.get_name()}: {result}")
                for objective_name, objective_value in result.items():
                    rows.append([benchmark, dataset_name, method, best_run_trial,
                                 metric.get_name(), objective_name, objective_value
                                 ])

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f"{output}.csv", index=False)

if __name__ == "__main__":
    evaluate_metrics()