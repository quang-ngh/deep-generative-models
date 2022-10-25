import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class Diffusion(nn.Module):
    def __init__(self, beta_start, beta_end, time_steps, sampling_steps, network):
        super(Diffusion, self).__init__()
        
        self.beta = torch.linspace(beta_start, beta_end, time_steps, device=device, requires_grad=False)
        self.alpha = 1.0 - self.beta
        self.cum_prod_alpha = torch.cumprod(self.alpha, dim = 0)
        self.one_minus_cumprod = 1.0 - self.cum_prod_alpha
        self.denoise_net = network
        self.sampling_steps = sampling_steps
        self.time_steps = time_steps
        
    def _posterior_sample(self, x, t):
        batch, c, h, w = x.shape
        cumprod_t = self.cum_prod_alpha[t].view(batch, 1, 1, 1)
        one_minus_cumprod_t = self.one_minus_cumprod[t].view(batch, 1, 1, 1)

        noise = torch.randn_like(x, device = device, requires_grad=False)
        std = torch.sqrt(one_minus_cumprod_t)
        mean = torch.sqrt(cumprod_t) * x

        return mean + std*noise, noise
    
    @torch.no_grad()
    def _reverse(self, noise, t):
        
        B, C, H, W= noise.shape
        z = torch.randn_like(noise) if t >= 1 else 0

        time = torch.ones(B, dtype=torch.int64, device=device)*t

        eps_theta = self.denoise_net(noise, time)
        eps_coff = (1.0-self.alpha[t]) / ((1-self.cum_prod_alpha[t])**0.5)

        x_previous = (1.0 / (self.alpha[t] ** 0.5)) * (noise - eps_coff * eps_theta) + z * ((1-self.alpha[t])**0.5)

        return x_previous
    
    @torch.no_grad()
    def sampling(self, image_shape: list, batch: int):
        
        C,H,W = image_shape
        image = torch.randn(batch, C, H, W, device = device, requires_grad=False)
        tracks = [image]

        t = self.sampling_steps - 1
        
        while t >= 0:
            image = self._reverse(image, t) #   Sample x_{t-1} from p(x_t-1|x_t)
            tracks.append(image)
            t-=1
        
        return image, tracks
    
    def forward(self, x, t):
        out, noise = self._posterior_sample(x, t)    # Diffuse data
        out = self.denoise_net(out, t)               # Predict noise
        B,C,H,W = x.shape
        oB, oC, oH, oW = out.shape
        assert (B,C,H,W) == (oB, oC, oH, oW), "Output shape is not compatible with input shape. Expect {} but {}".format((B,C,H,W),(oB, oC, oH, oW) )
        return out, noise