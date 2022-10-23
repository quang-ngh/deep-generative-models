from codecs import decode
import torch
import torch.nn as nn

class PyTDiscriminator(nn.Module):
    def __init__(self, decoder_net):
        super(PyTDiscriminator, self).__init__()
        self.net = nn.Sequential(*decoder_net)
    
    def forward(self, x):
        return self.net(x)