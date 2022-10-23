from re import S
import torchvision as vision
import torch
from pytorch.encoder import *
from pytorch.decoder import *
from pytorch.vanila import *
from pytorch.dataset import *
import matplotlib.pyplot as plt
import numpy as np

NVALUES = 256 
LATENT_DIM = 128
encoder_dims = [28*28, 512,256,LATENT_DIM]
decoder_dims = [LATENT_DIM,256,256,28*28*NVALUES]

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

torch.cuda.init()
torch.cuda.set_device(0)



def train_torchModel(epochs, learning_rate):
    
    model = PyTVAEs(encoder_dims, LATENT_DIM, decoder_dims, NVALUES, likelihood_type = "categorical")
    model.to("cuda")

    batch_size = 300
    train_loader = get_train_loader()
    
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        for (idx_batch, batch) in enumerate(train_loader):

            loss = model.forward(batch)
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
        

        if epoch % 100 == 0:
            
            print("Loss after {} epochs = {}".format(epoch, loss.item()))
            print("saved!")

            noise = torch.randn(16, LATENT_DIM).to("cuda")
            gen_img = model.decoder.sample(noise).cpu().detach()

            gen_img = gen_img.view(16,1,28,28)
            imgs = vision.utils.make_grid(gen_img, nrow=4)
            vision.utils.save_image(imgs, fp = "save\save_at_"+str(epoch) +".png")

if __name__ == '__main__':
    epochs = 1000
    learning_rate = 2e-4
    train_torchModel(epochs= epochs, learning_rate= learning_rate)