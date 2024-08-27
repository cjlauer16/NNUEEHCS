from torch import nn
from deltauq import deltaUQ_MLP, deltaUQ_CNN

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models])
        return outputs.mean(0)