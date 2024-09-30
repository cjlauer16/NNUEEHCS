import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.environments import LightningEnvironment

def _inst_init_if_not_none(inst, attr, val, default):
    if val is not None:
        setattr(inst, attr, val)
    else:
        setattr(inst, attr, default)


class Trainer(L.Trainer):
    def __init__(self, name, trainer_config, logger=None, callbacks=None, version=None, log_dir='logs'):
        self.name = name
        _inst_init_if_not_none(self, 'callbacks', 
                                callbacks, [EarlyStopping(monitor='val_loss')]
                                )
        _inst_init_if_not_none(self, 'logger', logger,
                               L.loggers.CSVLogger(log_dir, name=name, version=version)
                               )

        super().__init__(callbacks=self.callbacks, logger=self.logger, 
                         **trainer_config,
                         plugins=LightningEnvironment()
                         )
        self.logger.log_hyperparams(trainer_config)

    def get_logger(self):
        return self.logger

    def get_callbacks(self):
        return self.callbacks

    @classmethod
    def get_default_logdir(cls, dir, name, version):
        return L.loggers.CSVLogger(dir, name=name, version=version).log_dir
        


class ModelSavingCallback(L.callbacks.Callback):
    def __init__(self, monitor = "val_loss", save_path=None, model_name='model.pth'):
        self.monitor = monitor
        self.save_path = save_path
        self.model_name = model_name

    def on_fit_start(self, trainer, pl_module):
        self.trainer = trainer
        self.pl_module = pl_module

        if self.save_path is None:
            self.save_path = self.trainer.logger.log_dir

    def on_validation_end(self, trainer, pl_module):
        if self.monitor in trainer.callback_metrics:
            current = trainer.callback_metrics[self.monitor]
            if not hasattr(self, 'best'):
                self.best = current
                self.save_checkpoint(pl_module)
            elif current < self.best:
                self.best = current
                self.save_checkpoint(pl_module)

    def save_checkpoint(self, model):
        torch.save(model, self.save_path + f'/{self.model_name}')