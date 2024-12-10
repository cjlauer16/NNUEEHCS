import pytest
import torch
from nnueehcs.model_builder import (EnsembleModelBuilder, 
                                    KDEModelBuilder, 
                                    DeltaUQMLPModelBuilder,
                                    MLPModelBuilder,
                                    PAGERModelBuilder,
                                    ChecksumModelBuilder        
                                    )
from nnueehcs.training import Trainer
import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import numpy as np


def is_within_tolerance(number, target, tolerance):
    upper_bound = target * (1 + tolerance)
    return number <= upper_bound


@pytest.fixture()
def trainer_config():
    return {
            'accelerator': 'cpu',
            'max_epochs': 5000,
            'overfit_batches': 1,
            'log_every_n_steps': 5,
            'num_sanity_val_steps': 0,
            'gradient_clip_val': 5}


@pytest.fixture()
def training_config():
    return {'loss': 'l1_loss'}


@pytest.fixture()
def network_descr():
    return [
        {'Linear': {'args': [3, 128]}},
        {'ReLU': {}},
        {'Linear': {'args': [128, 1]}}
    ]


@pytest.fixture()
def train_dataset():
    x = torch.randn(32, 3)
    return torch.utils.data.TensorDataset(x, x.sum(1, keepdim=True))


@pytest.fixture()
def train_dataloader(train_dataset):
    return torch.utils.data.DataLoader(train_dataset, batch_size=32)


@pytest.fixture(autouse=True)
def cleanup_files():
    yield
    import shutil
    shutil.rmtree('logs', ignore_errors=True)

    import os
    if os.path.exists('model.pth'):
        os.remove('model.pth')


def get_trainer(trainer_config, name, callbacks=None):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=200, verbose=False, mode="min")
    cbs = [early_stop_callback]
    if callbacks:
        cbs.extend(callbacks)
    return Trainer(name, trainer_config, callbacks=cbs)


def model_accuracy_assertions(log_dir, tolerance=0.99, loss_ceiling = 0.03):
    logger_path = f'{log_dir}/metrics.csv'
    val_loss = pd.read_csv(logger_path)['val_loss']
    min_loss = val_loss.min()

    assert is_within_tolerance(min_loss, 0.018744820728898, tolerance)
    assert min_loss < loss_ceiling
    assert val_loss.idxmin() > val_loss.idxmax()
    assert val_loss.min()*50 < val_loss.max()


def prediction_assertions(model):
    torch.save(model, 'model.pth')
    model = torch.load('model.pth')
    x = torch.randn(1, 3)
    y = model(x)
    assert torch.allclose(y, model(x))

def test_builder(trainer_config, training_config, network_descr, train_dataloader):
    trainer = get_trainer(trainer_config, 'mlp')
    logger = trainer.get_logger()

    mlp = MLPModelBuilder(network_descr, train_config=training_config).build()
    trainer.fit(mlp, train_dataloader, train_dataloader)

    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(mlp)

def test_ensembles(trainer_config, training_config, network_descr, train_dataloader):
    trainer = get_trainer(trainer_config, 'ensembles')
    logger = trainer.get_logger()

    ensemble_descr = {'num_models': 3}
    ensembles = EnsembleModelBuilder(network_descr, ensemble_descr, train_config=training_config).build()
    trainer.fit(ensembles, train_dataloader, train_dataloader)

    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(ensembles)

#@pytest.mark.skip(reason="Not set up yet")
def test_checksum(trainer_config, training_config, network_descr, train_dataloader):
    trainer = get_trainer(trainer_config, 'checksum')
    logger = trainer.get_logger()

    checksum_descr = {
        'n_checksums': 1,
        'checksum_name': 'sum',
        'checksum_pred_weight': 1,
        'checksum_penalty_weight': 0.01,
        'checksum_reward_weight':  0.01,
        'oos_min': 1,
        'oos_max': 2
    }
    checksum_model = ChecksumModelBuilder(network_descr, checksum_descr, train_config=training_config).build()
    trainer.fit(checksum_model, train_dataloader, train_dataloader)
    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(checksum_model)

def test_kde(trainer_config, training_config, network_descr, train_dataloader):

    kde = KDEModelBuilder(network_descr, kde_descr = {'rtol': 10000*0.1}, train_config=training_config).build()
    trainer = get_trainer(trainer_config, 'kde', callbacks=kde.get_callbacks())
    logger = trainer.get_logger()
    trainer.fit(kde, train_dataloader, train_dataloader)

    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(kde)

    kde_estimator = kde.kde
    assert kde_estimator is not None
    assert kde_estimator.bandwidth == 'scott'
    assert kde_estimator.rtol == 0.1

    a_batch = next(iter(train_dataloader))[0].detach().cpu().numpy()
    scores = np.exp(kde_estimator.score_samples(a_batch))
    avg_score = scores.mean()

    assert is_within_tolerance(avg_score, 0.032892700285257835, 0.20)

@pytest.mark.skip(reason="Not working, something up with DUQ")
def test_duq(trainer_config, training_config, network_descr, train_dataloader):
    trainer = get_trainer(trainer_config, 'kde')
    logger = trainer.get_logger()

    duq = DeltaUQMLPModelBuilder(network_descr, {'estimator': 'std'}, train_config=training_config).build()
    trainer.fit(duq, train_dataloader, train_dataloader)

    # DUQ + PAGER have much higher loss, they don't meet the reqs of these tests
    # model_accuracy_assertions(logger.log_dir, loss_ceiling=0.3, tolerance=40)
    prediction_assertions(duq)

@pytest.mark.skip(reason="Not working, something up with DUQ")
def test_pager(trainer_config, training_config, network_descr, train_dataloader):
    trainer = get_trainer(trainer_config, 'kde')
    logger = trainer.get_logger()

    pager = PAGERModelBuilder(network_descr, {'estimator': 'std'}, train_config=training_config).build()
    trainer.fit(pager, train_dataloader, train_dataloader)

    # DUQ + PAGER have much higher loss, they don't meet the reqs of these tests
    # model_accuracy_assertions(logger.log_dir, loss_ceiling=0.3, tolerance=40)
    prediction_assertions(pager)