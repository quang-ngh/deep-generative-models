from tkinter import N
import torch
import torch.nn as nn
import numpy as np

class PyTEncoder(nn.Module):
    """
        Approxiamte the variational posterior p(z|x) 
        by DNN to estimating the (mean, std) vector of
        multivariate normal distribution
    """
    def __init__(self, encoder_dims, latent_dim):
        super(PyTEncoder, self).__init__()
        
        layers = []
        n_dims = len(encoder_dims)
        for idx in range(n_dims-1):
            layer = nn.Linear(encoder_dims[idx], encoder_dims[idx+1])
            activation = nn.LeakyReLU()
            layers.append(layer)
            layers.append(activation)
        
        self.net = nn.Sequential(*layers)
        self.mu_net = nn.Sequential(
            nn.Linear(encoder_dims[n_dims-1], latent_dim),
            nn.LeakyReLU()
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(encoder_dims[n_dims-1], latent_dim),
            nn.Softplus()
        )
        self.flatten = nn.Flatten(1,2) 
        self.PI = torch.from_numpy(np.asarray(np.pi))

    def encode(self, x):
        out = self.flatten(x)
        out = self.net(out)
        mu = self.mu_net(out)
        log_var = self.log_var_net(out)

        return mu, log_var

    def reparameterize(self, x, mu = None, log_var = None):
        if (mu is None) or (log_var is None):
            mu, log_var = self.encode(x)
        #Reparameterize trick
        std = torch.exp(0.5 * log_var) #var^2 = std
        eps = torch.randn_like(std)

        return mu + std * eps

    
    def sample(self, x):

        return self.reparameterize(x) 
        
    def log_prob(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(x, mu = mu, log_var = log_var)

        D = z.shape[1]
        log_p = -0.5*D*torch.log(2.*self.PI) - 0.5*log_var - 0.5 * (z - mu)**2 * torch.exp(-log_var)

        return log_p
        

    def forward(self, x, process = None):
    
        assert process in ["sample", "log_prob"], "TypeError!"

        if process == "sample":
            return self.sample(x)
        if process == "log_prob":
            return self.log_prob(x)