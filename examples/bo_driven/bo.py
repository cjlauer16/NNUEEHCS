import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
from nnueehcs.model_builder import (EnsembleModelBuilder, KDEModelBuilder, 
                                    DeltaUQMLPModelBuilder, PAGERModelBuilder, 
                                    MCDropoutModelBuilder)
from nnueehcs.training import Trainer, ModelSavingCallback
from nnueehcs.data_utils import get_dataset_from_config
from nnueehcs.evaluation import get_uncertainty_evaluator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as L
from ax.service.ax_client import AxClient, ObjectiveProperties
import yaml
import click
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import pandas as pd


class OutputManager:
    def __init__(self, directory_prefix, benchmark_name, append_benchmark_name=True):
        self.benchmark_name = benchmark_name
        if append_benchmark_name:
            self.output_dir_name = f'{directory_prefix}_{benchmark_name}'
        else:
            self.output_dir_name = f'{directory_prefix}'
        self.output_dir_path = Path(self.output_dir_name)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_datetime_prefix(cls):
        return datetime.now().strftime("%Y-%m-%d")

    def save_optimization_state(self, optimization_step, ax_client, name="ax_client"):
        ax_client.save_to_json_file(f"{str(self.output_dir_path)}/{name}.json")
        dat = {'optimization_step': optimization_step}
        with open(f'{str(self.output_dir_path)}/{name}_optimization_step.json', 'w') as f:
            f.write(json.dumps(dat))

    def save_pareto_parameters(self, pareto_parameters, name="pareto_parameters"):
        with open(f'{str(self.output_dir_path)}/{name}.json', 'w') as f:
            f.write(pareto_parameters)

    def save_trial_results_df(self, trial_results_df, name="trial_results"):
        # print trail results index name
        print(trial_results_df.index.name)
        trial_results_df.to_csv(f"{str(self.output_dir_path)}/{name}.csv", index=True)

    def save_trial_results_dict(self, trial_results_dict, name="trial_results"):
        # nested dict: {trial_index: {trial_results}}
        # use the 'trial_index' as the index when convertingo to a dataframe
        # set the name of the index to 'trial'
        trial_results_df = pd.DataFrame.from_dict(trial_results_dict, orient='index')
        trial_results_df.index.name = 'trial'
        self.save_trial_results_df(trial_results_df, name)

    def get_output_dir(self):
        return self.output_dir_path


@dataclass
class BOParameterWrapper:
    parameter_space: list
    parameter_constraints: list
    objectives: dict
    tracking_metric_names: list

    def get_parameter_names(self):
        return [p['name'] for p in self.parameter_space]


def get_params(config):
    parm_space = config['parameter_space']
    if 'constraints' in config:
        constraints = config['parameter_constraints']
    else:
        constraints = []
    objectives = config['objectives']
    tracking_metric_names = config['tracking_metrics']

    objectives_l = dict()
    for c in objectives:
        if c['type'] == 'minimize':
            objectives_l[c['name']] = ObjectiveProperties(minimize=True)
        else:
            objectives_l[c['name']] = ObjectiveProperties(minimize=False)
    return BOParameterWrapper(parm_space, constraints, objectives_l,
                              tracking_metric_names
                              )


def get_trainer(trainer_config, name, model, ue_method, dataset, version=None, log_dir='logs'):
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.00, patience=30, verbose=False, mode='min'),
                 ModelSavingCallback(monitor='val_loss')]
    extra_cbs = model.get_callbacks()
    if extra_cbs:
        callbacks.extend(extra_cbs)
    return Trainer(f'{name}/{dataset}/{ue_method}', trainer_config, 
                   callbacks=callbacks, log_dir=log_dir, 
                   version=version
                   )


def get_id_datset_name(dataset_name):
    return dataset_name + '_id'


def get_ood_dataset_name(dataset_name):
    return dataset_name + '_ood'


def get_model_builder_class(uq_method):
    if uq_method == 'ensemble':
        return EnsembleModelBuilder
    elif uq_method == 'kde':
        return KDEModelBuilder
    elif uq_method == 'delta_uq':
        return DeltaUQMLPModelBuilder
    elif uq_method == 'pager':
        return PAGERModelBuilder
    elif uq_method == 'mc_dropout':
        return MCDropoutModelBuilder
    else:
        raise ValueError(f'Unknown uq method {uq_method}')


def build_model(model_cfg, uq_config, uq_method, train_cfg):
    builder_class = get_model_builder_class(uq_method)
    builder = builder_class(model_cfg['architecture'],
                            uq_config[uq_method],
                            train_config = train_cfg
                            )
    return builder.build()


def get_dataset(dataset_cfg, dataset_name, is_ood=False):
    if is_ood:
        ds_name = get_ood_dataset_name(dataset_name)
    else:
        ds_name = get_id_datset_name(dataset_name)
    dset = get_dataset_from_config(dataset_cfg, ds_name)
    return dset


def prepare_dset_for_use(dset, training_cfg, scaling_dset=None):
    ipt = dset.input
    opt = dset.output
    if scaling_dset is None:
        scale_ipt = ipt
        scale_opt = opt
    else:
        scale_ipt = scaling_dset.input
        scale_opt = scaling_dset.output

    if training_cfg['scaling'] is True:
        # do min-max scaling to get it to 0-1
        opt = (opt - scale_opt.min()) / (scale_opt.max() - scale_opt.min())
        dset.output = opt
        ipt = (ipt - scale_ipt.min()) / (scale_ipt.max() - scale_ipt.min())
        dset.input = ipt
    return dset


def evaluate(model, id_dset, ood_dset):
    id_ipt = id_dset.input.to(model.device)
    id_opt = id_dset.output.to(model.device)
    ood_ipt = ood_dset.input.to(model.device)
    ood_opt = ood_dset.output.to(model.device)

    indices = torch.randperm(id_ipt.size(0))
    indices_ood = torch.randperm(ood_ipt.size(0))

    id_ipt = id_ipt[indices][:20000]
    id_opt = id_opt[indices][:20000]
    ood_ipt = ood_ipt[indices_ood][:20000]
    ood_opt = ood_opt[indices_ood][:20000]

    eval = get_uncertainty_evaluator('wasserstein')
    return eval.evaluate(model, (id_ipt, id_opt), (ood_ipt, ood_opt))


@click.command()
@click.option('--benchmark')
@click.option('--uq_method')
@click.option('--dataset', type=click.Choice(['tails', 'gaps']))
@click.option('--output', type=click.Path(), help="Name of output directory")
def main(benchmark, uq_method, dataset, output):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        trainer_cfg = config['trainer']
        training_cfg = config['training']
        model_cfg = config['benchmarks'][benchmark]['model']
        dataset_cfg = config['benchmarks'][benchmark]['datasets']
        uq_config = config['uq_methods']
        bo_config = config['bo_config']
        bo_config.update(uq_config[uq_method])
        bo_config['parameter_space'] += training_cfg['parameter_space']

    bo_params = get_params(bo_config)
    del training_cfg['parameter_space']
    del uq_config[uq_method]['parameter_space']
    name = benchmark
    ax_client = AxClient()
    ax_client.create_experiment(name="UE Tuning",
                                parameters=bo_params.parameter_space,
                                objectives=bo_params.objectives,
                                tracking_metric_names=bo_params.tracking_metric_names,
                                outcome_constraints=bo_params.parameter_constraints
                               )
    trial_results = dict()
    for bo_trial in range(bo_config['trials']):
        trial, index = ax_client.get_next_trial()
        lr = trial['learning_rate']
        bs = trial['batch_size']
        wd = trial.get('weight_decay', 0.0)
        del trial['learning_rate']
        del trial['batch_size']
        if 'weight_decay' in trial:
            del trial['weight_decay']
        training_cfg['learning_rate'] = lr
        training_cfg['batch_size'] = bs
        training_cfg['weight_decay'] = wd
        uq_config[uq_method].update(trial)

        dset = get_dataset(dataset_cfg, dataset)
        dset = prepare_dset_for_use(dset, training_cfg)
        model = build_model(model_cfg, uq_config, uq_method, training_cfg).to(dset.dtype)
        trainer = get_trainer(trainer_cfg, name, model, uq_method, dataset,
                              version=f'bo_trial_{bo_trial}',
                              log_dir=output)
        opt_manager = OutputManager(trainer.logger.log_dir, benchmark, append_benchmark_name=False)

        train_dl = DataLoader(dset, batch_size=training_cfg['batch_size'], shuffle=True)
        test_dl = DataLoader(dset, batch_size=training_cfg['batch_size'], shuffle=False)
        trainer.fit(model, train_dl, test_dl)

        model = torch.load(f'{trainer.logger.log_dir}/model.pth')

        model.eval()
        with torch.no_grad():
            dset_id = get_dataset(dataset_cfg, dataset)
            dset_ood = get_dataset(dataset_cfg, dataset, is_ood=True)

            # we have to scale ood relative to ID
            # we MUST scale OOD first because this method
            # modifies in place
            dset_ood = prepare_dset_for_use(dset_ood, training_cfg,
                                            scaling_dset=dset_id)
            dset_id = prepare_dset_for_use(dset_id, training_cfg)

            results = evaluate(model, dset_id, dset_ood)
            print(results)

            id_ue = results['id_ue']
            ood_ue = results['ood_ue']
            ue_time = np.array(results['id_time'] + results['ood_time'])
            ue_mean = ue_time.mean()
            ue_std = ue_time.std()

            unc_dist = np.array(results['uncertainty_distance'])
            unc_dist_mean = unc_dist.mean()
            unc_dist_std = unc_dist.std()

            trial_result = {#'ue_time': (ue_mean, ue_std),
                            'score_dist': (unc_dist_mean, unc_dist_std)}
            ax_client.complete_trial(trial_index=index, raw_data=trial_result)
            trial_results[index] = dict()
            trial_results[index].update(trial)
            trial_results[index]['ue_time'] = ue_mean
            trial_results[index]['learning_rate'] = lr
            trial_results[index]['score_dist'] = unc_dist_mean
            trial_results[index]['id_ue'] = id_ue.mean()
            trial_results[index]['ood_ue'] = ood_ue.mean()
            trial_results[index]['id_loss'] = results['id_loss']
            trial_results[index]['ood_loss'] = results['ood_loss']
            trial_results[index]['id_time'] = np.mean(results['id_time'])
            trial_results[index]['ood_time'] = np.mean(results['ood_time'])
            trial_results[index]['log_path'] = f'{trainer.logger.log_dir}'
            print(trial_results)
            fig, ax = plt.subplots()
            if isinstance(id_ue.data, tuple):
                firstdata_id = id_ue.data[0]
                firstdata_ood = ood_ue.data[0]

                seconddata_id = id_ue.data[1]
                seconddata_ood = ood_ue.data[1]
            else:
                firstdata_id = id_ue.data
                firstdata_ood = ood_ue.data

                seconddata_id = id_ue.data
                seconddata_ood = ood_ue.data

            ax.ecdf(firstdata_id.flatten(), label='ID')
            ax.ecdf(firstdata_ood.flatten(), label='OOD')
            ax.legend()
            # save it to png file
            plt.savefig(f'{trainer.logger.log_dir}/uncertainty.png')
            plt.clf()
            fig, ax = plt.subplots()

            ax.ecdf(seconddata_id.flatten(), label='ID')
            ax.ecdf(seconddata_ood.flatten(), label='OOD')
            ax.legend()
            # save it to png file
            plt.savefig(f'{trainer.logger.log_dir}/uncertainty_1.png')

        opt_manager.save_trial_results_dict(trial_results)
        opt_manager.save_optimization_state(index, ax_client)

    pareto_results = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)
    pareto_predictions = ax_client.get_pareto_optimal_parameters(use_model_predictions=True)
    pareto = {'results': pareto_results, 'predictions': pareto_predictions}
    opt_manager.save_pareto_parameters(json.dumps(pareto))

if __name__ == '__main__':
    main()
