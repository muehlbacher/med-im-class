# Medical Image Classification

Model was build and optimized on colab.research.google.com
For a better overview it was packed in subfiles and published on github.

This project was developed as part of a challenge in the master program "Artificial Intelligence" @ Johannes Kepler University (JKU) Linz


## Description

The goal of this challenge is the classification (Multiclass) of 9 regularly misidentified cell lines based on microscopy images. The classes to distinguish are PC-3, U-251 MG, HeLa, A549, U-2 OS, MCF7, HEK 293, CACO-2 and RT4.
The training data consists of 28.896 gray-scaled-images. Each sample contains at least one cell and consists of 3 seperate images showing different parts (nucleus, microtubules, endoplasmic reticulum) of the same cell. The separate images are combined to one 3-channel-image, that allows to treat them as a regular RGB image.

The performance of the Model is measured with balanced accuracy.

## Architecture

The best performance was reached with the pretrained ConvNeXt[^1][^2] Architecture with Base Weights. On top of the Convolutional layers there is a simple classifier stacked, a Linear Layer with Dropout and a ReLU activation function (this is not the optimal solution, you can play around to find the optimal output classifier).
The pretrained CNN Layer is locked and only the classifier is trained on the images. This results in better performance at the training process.
The ConvNeXt is used because it had the best results in the baseline model in comparison to other pretrained models (VGG16, DenseNet, AlexNet, ...). There is still room for hyperparameter tweaking to get better results. 

**The architecture.py is only there to get a view on the ConvNeXt architecture, during training the torchvision.models.convnext_base(...) is used**


## How to run

1. Clone your repository on your local system
2. Install relevent dependencies
3. modify the working_config.json
4. Run: `python main.py working_config.json`

[^1]: ConvNeXt Pytorch documentation: https://pytorch.org/vision/stable/models/convnext.html
[^2]: Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining. A ConvNet for the 2020s. arXiv:2201.03545, 2022; https://arxiv.org/abs/2201.03545