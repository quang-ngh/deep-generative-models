import torchvision
from torchvision.utils import transforms, make_grid, save_image
import imageio
import numpy as np

def save_and_display(grid: torch.Tensor, nrow: int, path: str) -> None:

    grid = make_grid(grid, nrow = nrow)
    save_image(grid, path)
    return grid, path

def save_gif(list_of_grids: list, path: str) -> None:

   toImg = transforms.ToPILImage()

   #  Each image is [4, c, h, w]
   images = [make_grid(x, nrow=4) for x in list_of_grids]
   gif_src = [np.array(toImg(img)) for img in images]

   imageio.mimsave(path, gif_src, format = 'GIF', fps = 10)