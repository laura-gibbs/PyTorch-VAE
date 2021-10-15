import glob
import os
# from models import vae
# from torchvision.datasets import MNIST
# from torchvision import DataLoader
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision.transforms import ToTensor
from PIL import Image
import torch


class CSDataset(Dataset):

    def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.paths = glob.glob(os.path.join(root_dir, '*.png'))
            print(root_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        image = Image.open(img_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(0)

