import torch.nn as nn
    
class SimpleCnnHeader(nn.Module):
    def __init__(self, input_dim, hidden_dims=[]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class SimpleCnn(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
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
    def __init__(self, input_dim=16*5*5, hidden_dims=[120, 84], out_dim=256, n_classes=10) -> None:
        super().__init__()

        self.base = SimpleCnnHeader(input_dim=input_dim, hidden_dims=hidden_dims)
        n_features = 84

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