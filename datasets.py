from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np


class ImageDataSet(Dataset):
    def __init__(self, image_dir, target_file):
        self.im_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
        self.targets = np.array(pd.read_csv(target_file))
    def __getitem__(self, idx):
        image = Image.open(self.im_files[idx])
        target = self.targets[idx]
        return image, target, idx
    def __len__(self):
        return len(self.im_files)

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

        rotated_image_data = TF.rotate(image_data, angle=45,
                                       interpolation=TF.InterpolationMode.BILINEAR)

        image = TF.to_tensor(rotated_image_data)

        return image, target
    
    def __len__(self):
        return len(self.dataset)
