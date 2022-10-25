import torch.nn as nn
import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

#   Sinoisudal Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):

        super(TimeEmbedding, self).__init__()
        self.dim = embed_dim

    def forward(self, t: torch.Tensor):

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim = -1)

        return emb 

#  Residual Block
class ResBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, num_groups: int, dropout_rate: float, down_up: str):
        super(ResBlock, self).__init__()

        self.swish = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = (3,3), padding = 'same', stride = 1, device=device)
        self.conv2 = nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = (5,5), padding = 'same', stride = 1, device= device)
        self.res_conv = nn.Conv2d(in_channels = in_channel, out_channels=out_channel, kernel_size=(7,7), padding = 'same', stride = 1, device=device)

        self.group_norm1 = nn.GroupNorm(num_groups=num_groups, num_channels = in_channel, device = device)
        self.group_norm2 = nn.GroupNorm(num_groups=num_groups, num_channels = out_channel, device = device)
        self.mode = down_up
        if down_up == "down":
            self.time_embedding = TimeEmbedding(out_channel)
            self.time_mlp = nn.Sequential(
                nn.Linear(out_channel, out_channel, device=device),
                nn.SiLU(),
            )
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
            X -> GroupNorm1 -> Swish -> Conv1 -> Out1
            Time embedding -> Swish -> MLP -> Out2
            (Out1 + Out2) -> GroupNorm2 -> Swish -> Dropout -> Conv2 -> Skip connection
        """
        B, C, H, W = x.shape

        out = x

        # X -> GroupNorm -> Swish -> Conv1 -> Out1
        out = self.group_norm1(out)
        out = self.swish(out)
        out = self.conv1(out)
        assert (H,W) == (out.shape[2], out.shape[3]), "Not compatible shape"

        # Time Embedding in the case of downblock
        if self.mode == "down":
            time_embed = self.time_embedding(t)
            time_embed = self.time_mlp(time_embed)
            time_embed = time_embed.view(B, time_embed.shape[1], 1, 1)
            out += time_embed
            
        #   Last
        out = self.group_norm2(out)
        out = self.swish(out)
        out = self.conv2(out)
        out += self.res_conv(x)

        # assert out.get_device() == device, "Not match devices"
        return out

#   Downsample Block
class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels = out_channel, kernel_size=(3,3), padding = 'same', stride = 1, device=device)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(4,4), padding = (1,1), stride = 2, device=device)
        self.norm = nn.BatchNorm2d(out_channel, device = device)
    
    def forward(self, x, t = None):
        B,C,H,W = x.shape    
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.activation(out)
        spatial_size = out.shape
        assert (spatial_size[2], spatial_size[3]) == (H//2, W//2), "Not compatible size!"
        return out

#   Upsample Block
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels = out_channel, out_channels = out_channel, kernel_size=(4,4), padding =1, stride = 2, device=device)
        self.norm = nn.BatchNorm2d(out_channel, device = device)
        
    def forward(self, x, t = None):
        B,C,H,W = x.shape
        out = self.deconv(x)
        out = self.norm(out)
        # assert out.get_device() == device, "Not match devices"
        assert (out.shape[2], out.shape[3]) == (H*2, W*2), "Size is not compatible!"
        return out

class Unet(nn.Module):
    """
    Residual Block:
        4 resolutions (256, 128, 64, 32)
        4 blocks per resolutions. In and out channels will be defined as 
        256 x 256: (3,6) -> (6,6) -> (6,6) -> (6,6)
        128 x 128: (6,12) -> (12,12) -> (12,12) -> (12,12)
        64 x 64: (12,15) -> (15,15) -> (15,15) -> (15,15)
        32 x 32: (15,18) -> (18,18) -> (18, 18) -> (18,18)
    
    Downsample Block:
        256 -> 128 -> 64 -> 32
        (6,6) -> (12,12) -> (15,15) -> (18,18)
    Upsample Block:
        32 -> 64 -> 128 -> 256
        (18, 15) -> (15,12) -> (12,6) -> (6,3)
    """
    def __init__(self, resolutions:list, in_channels: list, out_channels: list, num_groups: int, image_size: tuple):
        super(Unet, self).__init__()
        
        assert len(resolutions) == len(in_channels), "Given {} resolutions in down sampling but just have {} expected in channels".format(len(resolutions), len(in_channels))
        self.down_residual = nn.ModuleList([nn.ModuleList([]) for idx in range(len(resolutions))])
        self.midsample = nn.ModuleList([])
        self.upsample = nn.ModuleList([nn.ModuleList([]) for idx in range(len(resolutions))])
        self.downsample = nn.ModuleList([])
        self.downsample_res = nn.ModuleList([])
        
        in_out = list(zip(in_channels, out_channels))
        # Setting for the down-sample and upsample blocks
        for idx, pair_res in enumerate(in_out):
            
            in_channel, out_channel = pair_res
            for resblock_idx in range(4):
                in_channel_down = in_channel if resblock_idx == 0 else out_channel
                out_channel_down = out_channel
                self.down_residual[idx].append(
                    ResBlock(in_channel_down, out_channel_down, num_groups, 0.0, down_up = "down")
                )
                
                in_channel_up = out_channel * 2 if resblock_idx == 0 else in_channel
                out_channel_up = in_channel
                self.upsample[idx].append(
                    ResBlock(in_channel_up, out_channel_up, num_groups, 0.0, down_up = "up")
                )
            self.downsample.append(
                DownBlock(out_channel_down, out_channel_down)
            )
            self.downsample_res.append(
                DownBlock(out_channel_down, out_channel_down)
            )
            self.upsample[idx].append(
                UpBlock(out_channel_up, out_channel_up)
            )
        self.upsample = self.upsample[::-1]
        
        self.midsample.append(nn.Conv2d(out_channels[-1], out_channels[-1], kernel_size=(3,3), padding = 'same', stride = 1, device = device))
        self.midsample.append(nn.Identity())
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.resolutions = resolutions
        
    def forward(self, x, t):
        
        """
        ResBlock (n blocks)
            :input: x: batch of images, t: time steps
        DownBlock:
            :input: Output of ResBlock
            256 x 256 -> 32 x 32
        MiddleBlock:
            :input: Output of Downblock -> Size / 8
            32 x 32 -> 16 x 16
        UpBlock:
            :input: Output of MiddleBlock
            16 x 16 -> 256 x 256
        """
        B,C,H,W = x.shape
        out = x
        n_resolutions = len(self.resolutions)
        resolution_down = []
        residual_down = []
        for down_idx, downblock in enumerate(self.down_residual):
            
            # Pass through the residual blocks and down sample 
            for dblock in downblock:
                out = dblock(out, t)
            out = self.downsample[down_idx](out, t)
            dB, dC, dH, dW = out.shape
            scale_down_factor = 2 ** (down_idx + 1)
            assert (dB, dC, dH, dW) == (B, self.out_channels[down_idx], H //scale_down_factor, W // scale_down_factor),\
            "Shape is not compatible. Expected outshape of {} but {}".format((B, self.out_channels[down_idx], H //scale_down_factor, W // scale_down_factor), (dB, dC, dH, dW))
            resolution_down.append(out)

        # Mid sample block: 16 x 16 -> 16 x 16
        for mid_block in self.midsample:
            out = mid_block(out)
            
        # Passing through the residual blocks and up sample
        for up_idx, upblock in enumerate(self.upsample):
            out = torch.cat((out, resolution_down.pop()), dim = 1)
            for ublock in upblock:
                out = ublock(out, t)
            uB, uC, uH, uW = out.shape
            # assert (uB, uC, uH, uW) == (B, self.in_channels[::-1][up_idx], self.resolutions[n_resolutions - up_idx - 1], self.resolutions[n_resolutions - up_idx - 1]),\
            # "Shape is not compatible. Expected outshape of {} but {}".format((B, self.in_channels[::-1][up_idx], self.resolutions[n_resolutions - up_idx - 1], self.resolutions[n_resolutions - up_idx - 1]), (uB, uC, uH, uW))
        
        oB, oC, oH, oW = out.shape
        assert (oB, oC, oH, oW) == (B, C, H, W), "Output shape is not compatible. Expect {} but {}".format((B,C,H,W), (oB, oC, oH, oW))
        return out