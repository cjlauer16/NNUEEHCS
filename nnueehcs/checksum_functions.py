import torch
from torch import nn

class Checksum(torch.nn.Module):
    def __init__(self, min_enforce = 0, max_enforce=1.e16, eps = 1.e-10):
        super(Checksum, self).__init__()
        self.min_enforce = min_enforce
        self.max_enforce = max_enforce
        self.eps = eps

    def calc_checksum(self, y):
        print('Replace this with a specific child class')
        raise(AssertionError)
        return
    
    def calc_checksum_mse(self, y_true, checksum_pred):
        checksum_true = self.calc_checksum(y_true)
        return nn.MSELoss()(checksum_pred, checksum_true)
    
    def checksum_err_penalty(self, y_pred):
        checksum_pred = y_pred[:,-1]
        checksum_target = self.calc_checksum(y_pred[:,:-1])
        return nn.MSELoss()(checksum_pred, checksum_target)

    def checksum_err_reward(self, y_pred):
        ###### FIX ME: ###########
        # Change the summation over sum(1/err) not 1/(sum(err))
        # I think this is fixed now CL 12/6/24
        ##########################
        checksum_pred = y_pred[:,-1]
        checksum_target = self.calc_checksum(y_pred[:,:-1])
        denom = (checksum_pred - checksum_target)**2 + self.eps
        return torch.mean(1 / denom)
        #return 1 / (nn.MSELoss()(checksum_pred, checksum_target) + self.eps)
    
    def calc_pointwise_error(self, y_pred):
        checksum_pred = y_pred[:,-1]
        checksum_target = self.calc_checksum(y_pred[:,:-1])
        return torch.abs(checksum_pred - checksum_target)

class SummationChecksum(Checksum):
    def __init__(self, min_enforce = 0, max_enforce=1.e16, eps = 1.e-10):
        super().__init__(min_enforce, max_enforce, eps)

    def calc_checksum(self, y):
        return torch.sum(y, axis=1)

class SineChecksum(Checksum):
    def __init__(self, freq, min_enforce = 0, max_enforce=1.e16, eps = 1.e-10):
        super().__init__(min_enforce, max_enforce, eps)
        self.freq = freq

    def calc_checksum(self, y):
        return torch.sin(self.freq * torch.abs(torch.sum(y, axis=1)))