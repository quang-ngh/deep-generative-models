import torch.nn as nn
import torch 
import numpy as np

class Prior(nn.Module):
    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L
        self.PI = torch.from_numpy(np.asarray(np.pi))

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, x, reduction = None, dim= None):
        D = x.shape[1]
        log_p = -0.5 * D * torch.log(2. * self.PI) - 0.5 * x**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p