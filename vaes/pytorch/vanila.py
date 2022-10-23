import torch 
import torch.nn as nn
from .decoder import *
from .encoder import *
from .prior import *
import math
class PyTVAEs(nn.Module):
    
    def __init__(self, encoder_dims, latent_dim, decoder_dims, n_vals, likelihood_type):
        super(PyTVAEs, self).__init__()
        self.encoder = PyTEncoder(encoder_dims, latent_dim)
        self.decoder = PyTDecoder(decoder_dims, n_vals = n_vals, distribution = likelihood_type)
        self.prior = Prior(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):

        z = self.encoder(x, "sample")
        _x = self.decoder(x,z, "sample")

        reconstruction_loss = self.decoder(x,z,"log_prob")
        kl_loss = (-self.prior.log_prob(z) + self.encoder(x, "log_prob")).sum(-1)
        total_loss = -(reconstruction_loss - kl_loss)

        return total_loss.mean()
    
    def sample(self, batch_size):
        noise = torch.randn(batch_size, self.latent_dim).to("cuda")
        out = self.decoder.sample(noise)

        dim = int(math.sqrt(out.shape[1]))
        out = out.view(out.shape[0], dim, dim)
    
        return out

