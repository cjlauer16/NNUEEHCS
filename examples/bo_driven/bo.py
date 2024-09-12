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
    elif uq_method == 'delta_uq':
        return DeltaUQMLPModelBuilder
    elif uq_method == 'pager':
        return PAGERModelBuilder
    else:
        raise ValueError(f'Unknown uq method {uq_method}')


def build_model(model_cfg, uq_config, uq_method):
    builder_class = get_model_builder_class(uq_method)
    builder = builder_class(model_cfg['architecture'],
                            uq_config[uq_method]
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
def main(benchmark, uq_method, dataset):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        trainer_cfg = config['trainer']
        training_cfg = config['training']
        model_cfg = config['benchmarks'][benchmark]['model']
        dataset_cfg = config['benchmarks'][benchmark]['datasets']
        uq_config = config['uq_methods']

    name = benchmark

    dset = get_dataset(dataset_cfg, dataset)
    dset = prepare_dset_for_use(dset, training_cfg)
    model = build_model(model_cfg, uq_config, uq_method).to(dset.dtype)
    trainer = get_trainer(trainer_cfg, name, model)

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

        fig, ax = plt.subplots()
        ax.ecdf(id_ue.data[0].flatten(), label='ID')
        ax.ecdf(ood_ue.data[0].flatten(), label='OOD')
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


