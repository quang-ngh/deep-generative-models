from train import *
import torch
import torch.nn as nn
import argparse
from dataset import *
from vanila.vanilaGAN import *

GPU = True if torch.cuda.is_available() else False

def main():

    #   Add training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 500)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--k_step", type=int, default =1)
    parser.add_argument("--latent_dim", type = int, default = 256)
    parser.add_argument("--input_shape, type = int", default =784)
    train_args = parser.parse_args()

    GENERATOR_DIM = [train_args.latent_dim, 256, 512, 784]
    DISCRIMINATOR_DIM = [784,512,256,1 ]

    #   Get dataset for training
    train_loader = get_train_loader(train_args.batch_size)
    #   Create generator networks and discriminator networks
    generator_nets = []
    discriminator_nets = []

    for idx in range(len(GENERATOR_DIM)-1):

        layer = nn.Linear(GENERATOR_DIM[idx], GENERATOR_DIM[idx+1])
        activation = nn.ReLU()
        generator_nets.append(layer)
        generator_nets.append(activation)
    
    for idx in range(len(DISCRIMINATOR_DIM)-1):

        layer = nn.Linear(DISCRIMINATOR_DIM[idx], DISCRIMINATOR_DIM[idx+1])
        
        if idx + 1 == len(DISCRIMINATOR_DIM)-1:
            activation = nn.Sigmoid()
        else:
            activation = nn.ReLU()
        discriminator_nets.append(layer)
        discriminator_nets.append(activation)

    #   Create model
    model = VanilaGAN(
        encoder_net = generator_nets, decoder_net = discriminator_nets,
        latent_dim = train_args.latent_dim, batch_size = train_args.batch_size
    )
    
    #   Train model 
    loss = torch.nn.BCELoss()
    if GPU:
        model.generator.to("cuda")
        model.discriminator.to("cuda")
        loss.to("cuda")

    train_Pytorch_GAN(
        model = model, loss = loss, batch_size = train_args.batch_size, k_step=train_args.k_step,
        epochs = train_args.epochs, lr = train_args.lr, train_loader = train_loader,
        val_loader = None
    )

main()
