import torch.nn as nn
import torch 
import matplotlib.pyplot as plt
from torch.autograd import Variable

def _save_imgs(x_gen, name):
    n_rows = 4 # number of rows 
    n_cols = 4 # number of columns

    fig, axis = plt.subplots(figsize = (8, 8), nrows=n_rows, ncols = n_cols)

    for (idx, ax) in enumerate(axis.flatten()):
        plottable_img = x_gen[idx].view(28,28).cpu().detach().numpy()
        ax.imshow(plottable_img, cmap ="gray")
        ax.axis("off")
    
    plt.savefig(name)

def train_Pytorch_GAN(
    model, loss, batch_size, 
    epochs, k_step, lr,
    train_loader, val_loader = None
):
    
    ### Optimizer for generator and discriminator ### 
    G_optimizer = torch.optim.Adam(model.generator.parameters(), lr = lr)
    D_optimizer = torch.optim.Adam([p for p in model.discriminator.parameters() if p.requires_grad == True], lr = lr)


    real = Variable(torch.ones(batch_size, 1), requires_grad= False).to("cuda")
    fake = Variable(torch.zeros(batch_size, 1), requires_grad=False).to("cuda")

    print("Starting training...")
    model.train()
    for epoch in range(epochs):

        ### Sample real images from dataset ###
        for (idx, real_images) in enumerate(train_loader):

            real_images = real_images.view(real_images.shape[0], real_images.shape[1]*real_images.shape[2]) /255.0
            #_save_imgs(real_images, "real.png")
            fake_images = Variable(model.generator.sample(), requires_grad = False) ### Sample mini-batch from prior N(0,1)
            
            if epoch % k_step == 0:
            

                D_images = torch.cat((real_images, fake_images), dim = 0)
                D_label = torch.cat((real, fake), dim = 0)
                #   Train Discriminator
                
                D_optimizer.zero_grad()
                d_loss = loss(model.discriminator(D_images), D_label)
                d_loss.backward()
                D_optimizer.step()
            
            #   Train Generator
            
            G_optimizer.zero_grad() ### Init
            
            g_loss = loss(model.discriminator(fake_images), real) ### Fake or Real ?
            g_loss.backward()
            G_optimizer.step()

            

        if epoch % 50 == 0:
            _gen_image = model.generator.sample() 
            _save_imgs(_gen_image, "IMG_after_"+str(epoch)+".png")
            print("[Epochs : {}]  Discriminator Loss = {} ------ Generator Loss = {} ".format(epoch, d_loss.item(), g_loss.item()))

    print("End of training...")