import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ThreeLayersHeader(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 64]):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ThreeLayers(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class SimpleCnnHeader(nn.Module):
    def __init__(self, input_dim, hidden_dims=[]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class SimpleCnn(nn.Module):
    def __init__(self, input_dim=400, hidden_dims=[120, 84], out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FedconNet(nn.Module):
    """
    According to the paper, the whole NN are contacted with 3 parts
    1. Base Coder(Header) which extract representative vector from samples
    2. Project Head
    3. Output Layer
    """

    def __init__(self, out_dim=256, n_classes=10, base="cnn") -> None:
        super().__init__()

        if base == "cnn":
            self.base = SimpleCnnHeader(input_dim=400, hidden_dims=[120, 84])
            n_features = 84
        elif base == "resnet18":
            self.base = nn.Sequential(*list(resnet18().children())[:-1])
            n_features = 512
        elif base == "fc":
            self.base = ThreeLayersHeader(input_dim=784, hidden_dims=[128, 64])
            n_features = 64


        self.proj = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, out_dim)
        )

        self.output = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        """
        :param x: sample
        :return: projected representative vector 'z', label 'y'
        """
        h = self.base(x)
        h = h.squeeze()
        z = self.proj(h)
        y = self.output(z)
        return z, y