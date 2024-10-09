import click
import torch
from torch.utils.data import DataLoader
from nnueehcs.data_utils import get_dataset_from_config
from nnueehcs.evaluation import get_uncertainty_evaluator
from nnueehcs.classification import PercentileBasedIdOodClassifier
import pytorch_lightning as L
from pytorch_lightning.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False
from scipy.stats import pearsonr

from collections import defaultdict
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def prepare_dset_for_use(dset, training_cfg, scaling_dset=None):
    ipt = dset.input
    opt = dset.output
    if scaling_dset is None:
        scale_ipt = ipt
        scale_opt = opt
    else:
        scale_ipt = scaling_dset.input
        scale_opt = scaling_dset.output

    if True or training_cfg['scaling'] is True:
        # do min-max scaling to get it to 0-1
        opt = (opt - scale_opt.min()) / (scale_opt.max() - scale_opt.min())
        dset.output = opt
        ipt = (ipt - scale_ipt.min()) / (scale_ipt.max() - scale_ipt.min())
        dset.input = ipt
    return dset

def get_final_bo_trial(result_files):
    import re
    number_re = re.compile(r'\d+')
    largest_trial = None
    largest_trial_num = None
    for filename in result_files:
        fname = str(filename.name)
        trial_num = int(number_re.search(fname).group())
        file_exists = Path(f'{filename}/trial_results.csv').exists()
        if (largest_trial is None or trial_num > largest_trial_num) and file_exists:
            largest_trial = filename
            largest_trial_num = trial_num
    return largest_trial_num, largest_trial


def get_dataset_name(path):
    parts = path.parts
    return (parts[-2], parts[-1])


def get_id_datset_name(dataset_name):
    return dataset_name + '_id'


def get_ood_dataset_name(dataset_name):
    return dataset_name + '_ood'


def get_dataset(dataset_cfg, dataset_name, is_ood=False):
    if is_ood:
        ds_name = get_ood_dataset_name(dataset_name)
    else:
        ds_name = get_id_datset_name(dataset_name)
    dset = get_dataset_from_config(dataset_cfg, ds_name)
    return dset


def do_classification(model, id_dset, ood_dset, threshold):
    classifier = PercentileBasedIdOodClassifier(threshold)

    id_ipt, id_opt = id_dset.input, id_dset.output
    ood_ipt, ood_opt = ood_dset.input, ood_dset.output

    return classifier.evaluate(model, (id_ipt, id_opt), (ood_ipt, ood_opt))

def plot_quantity(results, quantity, output, x_axis='trial', scatter=False):
    # one file for each benchmark/dataset
    # draw lines for the different methods
    for bench, datasets in results.items():
        for dataset, methods in datasets.items():
            fig, ax = plt.subplots()
            ax.set_title(f'{bench}/{dataset}')
            for method, data in methods.items():
                tresults = data
                if scatter:
                    r2 = pearsonr(tresults[x_axis], tresults[quantity]).statistic
                    ax.scatter(tresults[x_axis], tresults[quantity], label=method)
                    # round r2 to two places
                    r2 = round(r2, 2)
                    ax.set_title(f'{bench}/{dataset} R2: {r2}')
                else:
                    ax.plot(tresults[x_axis], tresults[quantity], label=method)
                ax.set_xlabel(x_axis.capitalize())
                ax.set_ylabel(quantity)
            ax.legend()
            fig.savefig(f'{output}/{bench}_{dataset}_{x_axis}_{quantity}.png')
            plt.close(fig)



@click.command()
@click.option('--input', type=click.Path(exists=True), help='Path to directoy containing experiment results')
@click.option('--config', type=str, help='Path to config file')
@click.option('--output', type=click.Path(), help='Path to output file')
@click.option('--percentile', '-p', type=float, default=0.8, help='Precentile score threshold for ID/OOD split.')
def main(input, config, output, percentile):
    torch.set_grad_enabled(False)
    indir = Path(input)
    benchmarks = list()
    datasets = list()
    methods = list()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    results = dict()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for benchdir in indir.iterdir():
        results[benchdir.name] = dict()
        bench_results = results[benchdir.name]
        for datasetdir in benchdir.iterdir():
            print(datasetdir)
            benchmark, dataset = get_dataset_name(datasetdir)
            ds_cfg = config['benchmarks'][benchmark]['datasets']
            ds_id = get_dataset(ds_cfg, dataset, is_ood=False).to(device)
            ds_ood = get_dataset(ds_cfg, dataset, is_ood=True).to(device)
            ds_ood = prepare_dset_for_use(ds_ood, None, scaling_dset=ds_id)
            ds_id = prepare_dset_for_use(ds_id, None)
            bench_results[datasetdir.name] = dict()
            dataset_results = bench_results[datasetdir.name]
            for methoddir in datasetdir.iterdir():
                dataset_results[methoddir.name] = dict()
                bo_files = list()
                model_files = list()
                for resultfile in methoddir.iterdir():
                    # bo_results.append(str(resultfile))
                    bo_files.append(resultfile)
                trial_num, trial_file = get_final_bo_trial(bo_files)
                tresults_csv = pd.read_csv(f'{trial_file}/trial_results.csv')
                for success in range(trial_num):
                    success_dir = Path(f'bo_trial_{success}')
                    success_dir = methoddir / success_dir
                    model_files.append(success_dir / Path('model.pth'))
                for trial, model in enumerate(model_files):
                    model = torch.load(model).to(device)
                    model.eval()
                    tresults = do_classification(model, ds_id, ds_ood, percentile)
                    # dataset_results[methoddir.name][trial] = tresults
                    tresults['trial'] = trial
                    tresults['ue_dist'] = tresults_csv.query('trial == @trial')['score_dist'].values[0]
                    print(trial, tresults)
                    dataset_results[methoddir.name][trial] = tresults
                df = pd.DataFrame.from_dict(dataset_results[methoddir.name], orient='index')
                df.index.name = 'trial'
                dataset_results[methoddir.name] = df

    plot_quantity(results, 'sensitivity', output)
    plot_quantity(results, 'sensitivity', output, x_axis='ue_dist', scatter=True)

if __name__ == '__main__':
    main()
