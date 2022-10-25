from tqdm import tqdm
import torch 
import torchvision 
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from core.model import Diffusion
from core.unet import Unet
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, data_loader, epochs, lr, save_per_epochs, clip_val, loss_type, save_path):

    optimizer = torch.optim.Adam(model.denoise_net.parameters(), lr=lr)
    print(optimizer)
    print("Starting training...")
    
    losses = []
    if loss_type == "l1":
      loss_fn = torch.nn.L1Loss(reduction = 'mean')
    elif loss_type == "l2":
      loss_fn = torch.nn.MSELoss(reduction = 'mean')
    for epoch in range(epochs):
        
        for (idx, dataset) in tqdm(enumerate(data_loader)):

            x_0 = dataset[0].to(device)
            t = (model.time_steps - 1) * torch.rand((x_0.shape[0],), device = device) + 1
            t = t.long()
            optimizer.zero_grad()
            eps_theta, eps = model(x_0, t)

            loss = loss_fn(eps_theta, eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoise_net.parameters(), clip_val)
            optimizer.step()        
            losses.append(loss.item())

        print("Loss after {} = {}".format(epoch, loss.item()))
        if epoch % save_per_epochs == 0:
            torch.save(model.state_dict(), save_path+"/ckpt_"+str(epoch))
            # visualize(model, 4, save_path +"/diff_"+str(epoch)+".png" )
            print('Saved!')

    print("End training!")
    plt.plot(losses)
    plt.savefig("/content/drive/MyDrive/Diffusion-Model/ckpts/loss.png")

def get_model(beta_start, beta_end, time_steps, sampling_steps, unet):

    model = Diffusion(beta_start, beta_end, time_steps, sampling_steps, unet)
    print("Init model successfully!")
    return model

def get_model_pretrain(beta_start, beta_end, time_steps, sampling_steps, unet, ckpt_path):
    model = Diffusion(beta_start, beta_end, time_steps, sampling_steps, unet)
    model.load_state_dict(torch.load(ckpt_path))
    print("Load model successfully!")
    return model

if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type = int, default = 1000)
    parse.add_argument('--lr', type=float, default = 0.0001)
    parse.add_argument('--save_per_epochs', type = int, default = 100)
    parse.add_argument('--loss_type', type=str, default = 'l2')
    parse.add_argument('--save_path', type=str)
    parse.add_argument('--beta_start', type = float, default = 1e-3)
    parse.add_argument('--beta_end', type = float, default =2e-1)
    parse.add_argument('--time_steps', type=int, default = 1000)
    parse.add_argument('--sampling_steps', type = int, default = 1000)
    parse.add_argument('--ckpt_path', type = str, default = "")
    parse.add_argument('--batch_size', type=int, default=64)
    train_args = parse.parse_args()
    
    in_c = [3,12,18]
    out_c = [12,18,24]
    resolutions = [32,16,8]
    unet = Unet(resolutions, in_c, out_c, 3, (32,32)).to(device)

    if not train_args.ckpt_path:
        model = get_model(
            train_args.beta_start, train_args.beta_end, train_args.time_steps, train_args.sampling_steps, unet
        )
    else:
        model = get_model_pretrain(
            train_args.beta_start, train_args.beta_end, train_args.time_steps, train_args.sampling_steps, unet, train_args.ckpt_path
        )
    
    SIZE = 32
    batch_size = train_args.batch_size
    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Lambda( lambda t: (t * 2) - 1)
    ])

    dataset = torchvision.datasets.CIFAR10(root = ".", train=False, transform = transform, download = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train(model, dataloader, train_args.epochs, train_args.lr, train_args.epochs, 0.1, train_args.loss_type, 'save/')