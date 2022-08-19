from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
import torch


class ImageDataSet(Dataset):
    def __init__(self, image_dir, target_file):
        self.im_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
        self.targets = np.array(pd.read_csv(target_file))
        self.image_dir = image_dir

    def __getitem__(self, idx):
        #target[0]: id image, target[1]: label
        target = self.targets[idx]

        #image string: 5-numbers, +target[0] + _ + blue/red/yellow

        image_string_blue = f"{target[0]:05d}"+ "_"+ "blue" + ".png"
        image_string_red = f"{target[0]:05d}"+ "_"+ "red" + ".png"
        image_string_yellow = f"{target[0]:05d}"+ "_"+ "yellow" + ".png"



        image_blue = np.array(Image.open(self.image_dir + "/" + image_string_blue))
        image_red = np.array(Image.open(self.image_dir + "/" + image_string_red))
        image_yellow = np.array(Image.open(self.image_dir + "/" + image_string_yellow))

        
        
        image = np.zeros((3, image_blue.shape[0], image_blue.shape[1]))

        image[0] = image_blue
        image[1] = image_red
        image[2] = image_yellow

        image = torch.Tensor(image)

        return image, target, idx
    def __len__(self):
        return len(self.targets)

class TransformImageDataset(Dataset):
    def __init__(self, dataset: Dataset, transforms=None, target_transform=None):
        self.dataset = dataset
        self.transforms = transforms
        self.target_transform = target_transform

    def __getitem__(self, idx):
        image_data, target, idx = self.dataset[idx]

        if self.transforms:
            image_data = self.transforms(image_data)

        if self.target_transform:
            target = self.target_transform(target)

        image = image_data
        return image, target
    
    def __len__(self):
        return len(self.dataset)
