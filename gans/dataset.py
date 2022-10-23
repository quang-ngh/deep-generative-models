import torch
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf

class MNIST_Digits(Dataset):

    def __init__(self, mode='train', transforms=None):
        (digits, _), _ = tf.keras.datasets.mnist.load_data()
        if mode == 'train':
            self.data = digits[:10000, :, :]    
        elif mode == 'val':
            self.data = digits[30000:32000, :, :]

        else:
            self.data = digits[32000:, :, :]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.Tensor(self.data[idx, : ,:]).to("cuda")
        if self.transforms:
            sample = self.transforms(sample)
        return sample

def get_train_loader(batch_size):
    train_data = MNIST_Digits(mode= "train")

    return DataLoader(train_data, batch_size=batch_size, shuffle = True)