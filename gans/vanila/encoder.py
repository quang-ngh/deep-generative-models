import torch
import torch.nn as nn

class PyTGenerator(nn.Module):
    def __init__(self, encoder_net, latent_dim, batch_size):
        
        super(PyTGenerator, self).__init__()
        self.net = nn.Sequential(*encoder_net)
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def generate(self, z):
        return self.net(z)

    def sample(self):
        ### Sample and generate fake image from the prior ###
        z = torch.randn(self.batch_size, self.latent_dim).to("cuda")
        return self.generate(z)
    
    def forward(self, x):

        if x is None:
            return self.sample()

        else:
            return self.generate(x)    
