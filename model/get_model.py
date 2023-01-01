import torch
from torchvision.models import resnet18, resnet101, resnet50
import torchvision
import torch.nn as nn


def get_model(model_name, pretrained=True):
    if model_name == "resnet18":
        net = torchvision.models.resnet18(pretrained=pretrained)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    if model_name == "resnet50":

        net = torchvision.models.resnet50(pretrained=pretrained)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    if model_name == "resnet101":

        net = torchvision.models.resnet101(pretrained=pretrained)

        # Replace 1st layer to use it on grayscale images
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    return net