import os 

import numpy as np
import torch

from architectures import MModel
from utils import classification, plot, trans_classification
from datasets import ImageDataSet, TransformImageDataset
from torchvision.transforms import Lambda
import torchvision.transforms as tf

def main(image_dir, target_file, results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 50_000, device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    image_dir = image_dir
    target_file = target_file

    image_data = ImageDataSet(image_dir,target_file)

    #
    transforms = tf.Compose([
                tf.RandomRotation(45)
                ])

    target_transform = Lambda(lambda y: trans_classification(y))

    transform_data = TransformImageDataset(image_data,transforms = transforms, target_transform = target_transform)
    i= 0
    for image, target in transform_data:
        print(image.shape)
        print(target)
        print(type(image))
        plot(image)
        if i >= 1:
            break
        i+=1

    # Prepare a path to plot to
    #plotpath = os.path.join(results_path, "plots")
    #os.makedirs(plotpath, exist_ok=True)

    # Create Network
    #net = MModel(**network_config)
    #net.to(device)

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
