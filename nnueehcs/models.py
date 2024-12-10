from torch import nn
from nnueehcs.deltauq import deltaUQ_MLP, deltaUQ_CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning as L
import copy
from nnueehcs.checksum_functions import *

training_defaults = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_workers': 1,
    'num_epochs': 10,
    'loss': 'l1_loss',
}


class WrappedModelBase(pl.LightningModule):
    def __init__(self, train_config=None,
                 validation_config=None
                 ):
        super(WrappedModelBase, self).__init__()
        self.train_config = copy.deepcopy(training_defaults)
        self.validation_config = copy.deepcopy(training_defaults)
        self.set_train_config(train_config)
        self.set_validation_config(validation_config)

    def set_train_config(self, train_config):
        if train_config is None:
            # We have a default train config,
            # we need to process it instead of
            # any user-defined overrides
            self.set_train_config(self.train_config)
            return

        self.train_config.update(train_config)
        self.loss = self.get_loss_fn(train_config['loss'])

    def set_validation_config(self, validation_config):
        if validation_config is None:
            # Yes, we default to the train config
            self.set_validation_config(self.train_config)
            return

        self.validation_config.update(validation_config)
        self.val_loss = self.get_loss_fn(validation_config['loss'])

    def get_loss_fn(self, name):
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown loss function: {name}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def on_train_start(self):
        all_params = {'train_config': self.train_config,
                      'validation_config': self.validation_config
                      }
        self.logger.log_hyperparams(all_params)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config['learning_rate'],
                                      weight_decay=self.train_config.get('weight_decay', 0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def get_callbacks(self):
        return []


class EnsembleModel(WrappedModelBase):
    def __init__(self, models, **kwargs):
        super(EnsembleModel, self).__init__(**kwargs)
        self.models = nn.ModuleList(models)

    def forward(self, x, return_ue=False):
        outputs = torch.stack([model(x) for model in self.models])
        if return_ue:
            std = outputs.std(0)
            return outputs.mean(0), std
        return outputs.mean(0)


class MCDropoutModel(WrappedModelBase):
    def __init__(self, model, num_samples=100, dropout_percent = 0.5, **kwargs):
        super(MCDropoutModel, self).__init__(**kwargs)
        self.model = model
        self.num_samples = num_samples
        self.dropout_perent = dropout_percent


    def forward(self, x, return_ue=False):
        preds = torch.stack([self.model(x) for _ in range(self.num_samples)])
        if return_ue:
            return preds.mean(0), preds.std(0)
        return preds.mean(0)


    def eval(self):
        super().eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

class MLPModel(WrappedModelBase):
    def __init__(self, model, **kwargs):
        super(MLPModel, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)


class KDEMLPModel(MLPModel):
    def __init__(self, base_model, bandwidth='scott', rtol=0.1, train_fit_prop=1.0, **kwargs):
        super(KDEMLPModel, self).__init__(base_model, **kwargs)
        self.bandwidth = bandwidth
        self.rtol = rtol/10000
        self.kde = None
        self.train_fit_prop = train_fit_prop

    def fit_kde(self, data):
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=self.bandwidth, rtol=self.rtol)
        # randomly select 'train_fit_prop' of the data
        train_idxes = torch.randperm(len(data))[:int(self.train_fit_prop * len(data))]
        train_data = data[train_idxes].detach().cpu().numpy()
        kde.fit(train_data)
        self.kde = kde


    def forward(self, x, return_ue=False):
        if return_ue and self.kde is None:
            raise ValueError("KDE not fitted yet")
        pred = super().forward(x)
        if return_ue:
            import time
            test = time.time()
            log_dens = self.kde.score_samples(x.detach().cpu().numpy())
            dens = torch.exp(torch.tensor(log_dens))
            tend = time.time()
            print(tend-test)
            return pred, dens
        return pred

    class KDEFitCallback(L.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._train_data_to_fit = []
            self._epochs = 0

        def on_train_epoch_end(self, trainer, pl_module):
            if self._epochs == 0:
                trn_data = torch.cat(self._train_data_to_fit)
                pl_module.fit_kde(trn_data)
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._epochs == 0:
                self._train_data_to_fit.append(batch[0])

    def get_callbacks(self):
        return [KDEMLPModel.KDEFitCallback()]


class DeltaUQMLP(deltaUQ_MLP, WrappedModelBase):
    def __init__(self, base_model, estimator='std', num_anchors=5, **kwargs):
        deltaUQ_MLP.__init__(self, base_model, estimator)
        # somehow, the constructor of WrappedModelBase
        # removes our 'net' member. We need to re-add it
        # after the initialization
        net = self.net
        WrappedModelBase.__init__(self, **kwargs)
        self.net = net
        self.num_anchors = num_anchors

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, torch.cat((y, y), dim=0))
        self.log('train_loss', loss)
        return loss

    def forward(self, x, return_ue=False):
        if self.training:
            return deltaUQ_MLP.forward(self, x)
        else:
            if not hasattr(self, 'anchors'):
                return deltaUQ_MLP.forward(self, x)
            return deltaUQ_MLP.forward(self, x, anchors=self.anchors, n_anchors=self.num_anchors, return_std=return_ue)

    @property
    def anchors(self):
        return self._anchors
    
    @anchors.setter
    def anchors(self, value):
        if not hasattr(self, '_anchors'):
            self.register_buffer('_anchors', value)
        else:
            self._anchors = value.detach().clone()

    class DeltaUQGetAnchorsCallback(L.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._train_data_to_fit = []
            self._epochs = 0

        def on_train_epoch_end(self, trainer, pl_module):
            if self._epochs == 0:
                trn_data = torch.cat(self._train_data_to_fit)
                pl_module.anchors = trn_data[0:pl_module.num_anchors].detach().clone()
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            bs = batch[0].shape[0]
            if self._epochs == 0 and bs*len(self._train_data_to_fit) < pl_module.num_anchors:
                self._train_data_to_fit.append(batch[0].detach())


    def get_callbacks(self):
        return [DeltaUQMLP.DeltaUQGetAnchorsCallback()]


class PAGERMLP(DeltaUQMLP, WrappedModelBase):
    def forward(self, x, return_ue=False):
        res = DeltaUQMLP.forward(self, x, return_ue)
        if not return_ue:
            return res

        pred, uncertainty = res
        conformal_scores = self._score_samples(x, self.anchors,
                                               self.anchors_Y)

        return pred, (uncertainty, conformal_scores)

    def _anchored_predictions(self, x, anchors):
        p_matrix = list()
        for sample in x:
            if len(sample.shape) == 1:
                sample = sample.unsqueeze(0)
            p = deltaUQ_MLP.forward(self,
                                    anchors,
                                    anchors=sample,
                                    n_anchors=len(sample),
                                    return_pred_matrix=True
                                    )
            p_matrix.append(p)
        return torch.concat(p_matrix).squeeze(-1)


    def _score_samples(self, x, anchors_X, anchors_Y):
        p_matrix = self._anchored_predictions(x, anchors_X)
        score = torch.max(torch.abs(p_matrix - anchors_Y.T), dim=1)[0]
        return score


    def get_callbacks(self):
        return [PAGERMLP.PAGERGetAnchorsCallback()]

    @property
    def anchors_Y(self):
        return self._anchors_Y
    
    @anchors_Y.setter
    def anchors_Y(self, value):
        if not hasattr(self, '_anchors_Y'):
            self.register_buffer('_anchors_Y', value)
        else:
            self._anchors_Y = value.detach().clone()

    class PAGERGetAnchorsCallback(L.callbacks.Callback):
        # Like DeltaUQGetAnchorsCallback, but we need
        # the input and outputs of the anchors
        def __init__(self):
            super().__init__()
            self._anchor_X = []
            self._anchor_Y = []
            self._epochs = 0

        def on_train_epoch_end(self, trainer, pl_module):
            if self._epochs == 0:
                nanchors = pl_module.num_anchors
                anchor_X = torch.cat(self._anchor_X)
                anchor_Y = torch.cat(self._anchor_Y)
                pl_module.anchors = anchor_X[0:nanchors].detach().clone()
                pl_module.anchors_Y = anchor_Y[0:nanchors].detach().clone()
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            bs = batch[0].shape[0]
            if self._epochs == 0 and bs*len(self._anchor_X) < pl_module.num_anchors:
                self._anchor_X.append(batch[0].detach())
                self._anchor_Y.append(batch[1].detach())


class ChecksumMLP(WrappedModelBase):
    def __init__(self, model, n_checksums=1, checksum_name='sum', freq=None,
                 checksum_pred_weight=1, checksum_penalty_weight=0, checksum_reward_weight=0,
                 oos_min=0.01, oos_max=0.05, **kwargs):
        super(ChecksumMLP, self).__init__(**kwargs)
        self.model = model
        self.num_outputs = model[-1].out_features
        self.n_checksums = n_checksums
        self.checksum_name = checksum_name
        self.checksum_pred_weight = checksum_pred_weight
        self.checksum_penalty_weight = checksum_penalty_weight
        self.checksum_reward_weight = checksum_reward_weight
        self.oos_min = oos_min
        self.oos_max = oos_max
        self.freq = freq   
        self.get_checksum_func(self.checksum_name)

        self.inputs_max = None
        self.inputs_min = None
        self.pre_set_min_max = False

    def get_checksum_func(self, name):
        if name == 'sum':
            self.Checksum = SummationChecksum()
        elif name == 'sine':
            if self.freq is None:
                raise ValueError("Frequency must be provided for sine checksum")
            self.Checksum = SineChecksum(self.freq)

    def set_input_bounds(self, x_min, x_max):
        self.inputs_min = x_min
        self.inputs_max = x_max
        self.pre_set_min_max = True
    
    def set_input_bounds_otf(self, x):
        if self.inputs_max is None:
            self.inputs_max = torch.max(x, axis=0).values
        else:
            self.inputs_max = torch.max(self.inputs_max, torch.max(x, axis=0).values)

        if self.inputs_min is None:
            self.inputs_min = torch.min(x, axis=0).values
        else:
            self.inputs_min = torch.min(self.inputs_min, torch.min(x, axis=0).values)

    def get_x_ood(self, n_ood, x_min, x_max, seed=None):
        diff = x_max - x_min
        upper_min = x_max + diff*self.oos_min
        upper_max = x_max + diff*self.oos_max
        lower_min = x_min - diff*self.oos_min
        lower_max = x_min - diff*self.oos_max

        if seed != None:
            torch.manual_seed(seed)
        mask = torch.randint(0, 2, diff.shape, dtype=torch.bool)
        if not mask.any():
            mask[torch.randint(0, mask.shape[0], size=(1,))] = True
        random_values = torch.rand(n_ood, diff.shape[0])

        assert(mask.any() == True)
        result = torch.where(mask,
                             upper_min + random_values*(upper_max-upper_min),
                             lower_min + random_values*(lower_max-lower_min))

        return result
    
    ###############################
    ## Rob's Implementation #########
    ###############################

    # def oos_data(self, batch_input):
    #    # get a random vector in the unit sphere.
    #    oos_sample = torch.randn_like(batch_input)
    #    #compute infinity norm
    #    oos_denom,_ = torch.max(torch.abs(oos_sample), dim=-1, keepdim=True)
    #    #oos_denom = oos_denom.expand_as(oos_sample)
    #    oos_sample = oos_sample / oos_denom

    #    center = (self.bb_max+self.bb_min)/2
    #    radius = (self.bb_max-self.bb_min)/2*1.05
    #    oos_sample = (oos_sample+center)*radius
    #    return oos_sample

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]

        # Make prediction
        y_pred = self(x)

        # calculate normal loss
        loss_val = self.loss(y, y_pred[:,:-1])
        total_loss = loss_val
        self.log('train_pred_loss', loss_val)

        if self.checksum_pred_weight > 0:
            checksum_loss = self.Checksum.calc_checksum_mse(y, y_pred[:,-1])
            total_loss = total_loss + self.checksum_pred_weight*checksum_loss/(self.num_outputs-1)
            self.log('train_checksum_loss', self.checksum_pred_weight*checksum_loss/(self.num_outputs-1))

        if self.checksum_penalty_weight > 0:
            checksum_penalty = self.Checksum.checksum_err_penalty(y_pred)
            total_loss = total_loss + self.checksum_penalty_weight * checksum_penalty
            self.log('train_checksum_penalty_loss', self.checksum_penalty_weight * checksum_penalty)

        if self.checksum_reward_weight > 0:
            if not self.pre_set_min_max:
                self.set_input_bounds_otf(x)
            x_ood = self.get_x_ood(batch_size, self.inputs_min, self.inputs_max)
            y_pred_ood = self(x_ood)
            checksum_reward = self.Checksum.checksum_err_reward(y_pred_ood)

            total_loss = total_loss + self.checksum_reward_weight * checksum_reward
            self.log('train_checksum_reward_loss', self.checksum_reward_weight * checksum_reward)

        self.log('train_loss', total_loss)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]

        # Make prediction
        y_pred = self(x)

        # calculate normal loss
        loss_val = self.loss(y, y_pred[:,:-1])
        total_loss = loss_val

        if self.checksum_pred_weight > 0:
            checksum_loss = self.Checksum.calc_checksum_mse(y, y_pred[:,-1])
            total_loss = total_loss + self.checksum_pred_weight*checksum_loss/(self.num_outputs-1)

        if self.checksum_penalty_weight > 0:
            checksum_penalty = self.Checksum.checksum_err_penalty(y_pred)
            total_loss = total_loss + self.checksum_penalty_weight * checksum_penalty

        if self.checksum_reward_weight > 0:
            if not self.pre_set_min_max:
                self.set_input_bounds_otf(x)
            x_ood = self.get_x_ood(batch_size, self.inputs_min, self.inputs_max)
            y_pred_ood = self(x_ood)
            checksum_reward = self.Checksum.checksum_err_reward(y_pred_ood)

            total_loss = total_loss + self.checksum_reward_weight * checksum_reward

        self.log('val_loss', total_loss)
        return total_loss


    def forward(self, x, return_ue=False):
        outputs = self.model(x)
        if return_ue:
            ue = self.Checksum.calc_pointwise_error(outputs)
            return outputs, ue
        else:
            return outputs