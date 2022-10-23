import torch
import torch.nn as nn
from .decoder import *
from .encoder import *

class VanilaGAN(nn.Module):
    
    def __init__(self, encoder_net, decoder_net, latent_dim,  batch_size):
        
        super(VanilaGAN, self).__init__()
        self.generator = PyTGenerator(encoder_net, latent_dim, batch_size)
        self.discriminator = PyTDiscriminator(decoder_net)
        self.EPS = 1.e-5
    
    def forward(self, x):
        pass
