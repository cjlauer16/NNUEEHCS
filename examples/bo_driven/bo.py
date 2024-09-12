import torch
from torch import nn
from torch.utils.data import DataLoader
from nnueehcs.model_builder import EnsembleModelBuilder, KDEModelBuilder, DeltaUQMLPModelBuilder, PAGERModelBuilder
from nnueehcs.training import Trainer, ModelSavingCallback
from nnueehcs.data_utils import get_dataset_from_config
from nnueehcs.evaluation import get_uncertainty_evaluator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as L
import yaml
import click
import matplotlib.pyplot as plt


def get_trainer(trainer_config, name, model):
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.00, patience=300, verbose=False, mode='min'),
                 ModelSavingCallback(monitor='val_loss')]
    extra_cbs = model.get_callbacks()
    if extra_cbs:
        callbacks.extend(extra_cbs)
    return Trainer(name, trainer_config, callbacks=callbacks)


def get_id_datset_name(dataset_name):
    return dataset_name + '_id'


def get_ood_dataset_name(dataset_name):
    return dataset_name + '_ood'


def get_model_builder_class(uq_method):
    if uq_method == 'ensemble':
        return EnsembleModelBuilder
    elif uq_method == 'kde':
        return KDEModelBuilder
    elif uq_method == 'delta':
        return DeltaUQMLPModelBuilder
    elif uq_method == 'pager':
        return PAGERModelBuilder
    else:
        raise ValueError(f'Unknown uq method {uq_method}')


@click.command()
@click.option('--benchmark')
@click.option('--uq_method')
@click.option('--dataset', type=click.Choice(['tails', 'gaps']))
def main(benchmark, uq_method, dataset):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        trainer_cfg = config['trainer']
        training_cfg = config['training']
        model_cfg = config['benchmarks'][benchmark]['model']
        dataset_cfg = config['benchmarks'][benchmark]['datasets']
        uq_config = config['uq_methods']

    name = benchmark
    train_ds_name = get_id_datset_name(dataset)

    dset = get_dataset_from_config(dataset_cfg, train_ds_name)
    builder_class = get_model_builder_class(uq_method)
    builder = builder_class(model_cfg['architecture'],
                            uq_config[uq_method]
                            )
    model = builder.build().to(dset.dtype)
    trainer = get_trainer(trainer_cfg, name, model)

    ipt = dset.input
    opt = dset.output
    # do min-max scaling to get it to 0-1
    opt = (opt - opt.min()) / (opt.max() - opt.min())
    dset.output = opt
    ipt = (ipt - ipt.min()) / (ipt.max() - ipt.min())
    dset.input = ipt

    train_dl = DataLoader(dset, batch_size=training_cfg['batch_size'], shuffle=True)
    test_dl = DataLoader(dset, batch_size=training_cfg['batch_size'], shuffle=False)
    trainer.fit(model, train_dl, test_dl)
    torch.save(model, 'model.pth')

    model = torch.load('logs/binomial_options/version_0/model.pth')

    model.eval()
    with torch.no_grad():
        id_dset_name = get_id_datset_name(dataset)
        ood_dset_name = get_ood_dataset_name(dataset)
        dset_id = get_dataset_from_config(dataset_cfg, id_dset_name)
        dset_ood = get_dataset_from_config(dataset_cfg, ood_dset_name)

        id_ipt = dset_id.input.to(model.device)
        id_opt = dset_id.output.to(model.device)

        ood_ipt = dset_ood.input.to(model.device)
        ood_opt = dset_ood.output.to(model.device)

        # We need to normalize opt before normalizing ipt
        ood_ipt = (ood_ipt - id_ipt.min()) / (id_ipt.max() - id_ipt.min())
        ood_opt = (ood_opt - id_opt.min()) / (id_opt.max() - id_opt.min())

        id_ipt = (id_ipt - id_ipt.min()) / (id_ipt.max() - id_ipt.min())
        id_opt = (id_opt - id_opt.min()) / (id_opt.max() - id_opt.min())

        indices = torch.randperm(id_ipt.size(0))
        indices_ood = torch.randperm(ood_ipt.size(0))

        id_ipt = id_ipt[indices][:20000]
        id_opt = id_opt[indices][:20000]
        ood_ipt = ood_ipt[indices_ood][:20000]
        ood_opt = ood_opt[indices_ood][:20000]

        eval = get_uncertainty_evaluator('wasserstein')
        results = eval.evaluate(model, (id_ipt, id_opt), (ood_ipt, ood_opt))

        print(results)

        id_ue = results['id_ue']
        ood_ue = results['ood_ue']

        fig, ax = plt.subplots()
        ax.ecdf(id_ue.data.flatten(), label='ID')
        ax.ecdf(ood_ue.data.flatten(), label='OOD')
        ax.legend()
        # save it to png file
        plt.savefig('uncertainty.png')
        plt.clf()
        fig, ax = plt.subplots()

        ax.ecdf(id_ue.data[1].flatten(), label='ID')
        ax.ecdf(ood_ue.data[1].flatten(), label='OOD')
        ax.legend()
        # save it to png file
        plt.savefig('uncertainty_1.png')





if __name__ == '__main__':
    main()


