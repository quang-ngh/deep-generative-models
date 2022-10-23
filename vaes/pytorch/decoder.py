import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F

class PyTDecoder(nn.Module):
    def __init__(self, decoder_dims, n_vals, distribution):
        super(PyTDecoder, self).__init__()
        
        layers = []
        n_dims = len(decoder_dims)

        for idx in range(n_dims-1):
            layer = nn.Linear(decoder_dims[idx], decoder_dims[idx+1])
            activation = nn.LeakyReLU()
            layers.append(layer)
            layers.append(activation)
        
        self.net = nn.Sequential(*layers)
        self.distribution = distribution
        self.vals = n_vals

    def decode(self, z):
        """
        Approximate the likelihood distribution p(x|z)
        """
        
        out = self.net(z)
        batch_size = out.shape[0]
        dim = out.shape[1] // self.vals

        if self.distribution == "categorical":
            out = out.view(batch_size, dim, self.vals)
            out = torch.softmax(out, dim = 2)
            return [out]

        if self.distribution == "bernoulli":
            out = torch.sigmoid(out)

            return [out]
    
    def sample(self, z):

        #DECODE - Approximate p(x|z)
        out = self.decode(z) #Shape (batch, dim, #values)
        out = out[0]
        
        if self.distribution == "bernoulli":
            return torch.bernoulli(out)

        if self.distribution == "categorical":
            batch = out.shape[0]
            dim = out.shape[1]
            out = out.view(out.shape[0], -1, self.vals)
            p = out.view(-1, self.vals)
            x_new = torch.multinomial(p, num_samples = 1).view(batch, dim)

            return x_new /255.0

    def log_prob(self, x, z, reduction = None, dim = None):
        
        p = self.decode(z)[0]
        EPS = 1e-10
        
        
        if self.distribution == "categorical":
            x = x.view(x.shape[0], x.shape[1]**2)
            x_one_hot = F.one_hot(x.long(), num_classes=256)
            log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
            if reduction == 'avg':
                return torch.mean(log_p, dim=dim)
            elif reduction == 'sum':
                return torch.sum(log_p, dim=dim)
            else:
                return log_p

        if self.distribution == "bernoulli":
            pp = torch.clamp(p, EPS, 1.-EPS)
            log_p = x*torch.log(pp) + (1.-x)*torch.log(1.-pp)
            if reduction == 'avg':
                return torch.mean(log_p, dim)
            elif reduction == 'sum':
                return torch.sum(log_p, dim)
            else:
                return log_p

    def forward(self,x, z, process):
        assert process in ["sample", "log_prob"], "TypeError"
        
        if process == "sample":
            return self.sample(z)
        
        if process == "log_prob":
            return self.log_prob(x,z,reduction = "sum", dim = -1).sum(-1)