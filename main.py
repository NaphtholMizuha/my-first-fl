import random
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
from fedlab.utils.dataset import CIFAR10Partitioner
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, SGD
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from tqdm import tqdm

device = 'cuda:0'

class FlServer(ABC):
    """
    abstract base class representing server in FL
    """

    @abstractmethod
    def aggregate(self, params):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


    @abstractmethod
    def set_params(self, w):
        pass


class FlClient(ABC):
    """
    abstract base class representing client in FL
    """
    def train(self):
        pass

    def get_params(self):
        pass

    def set_params(self, w):
        pass

class FedavgClient(FlClient):
    def __init__(self, model: nn.Module, dataloader: DataLoader, loss_fn):
        self.loss_fn = loss_fn
        self.model = model
        self.dataloader = dataloader
        self.optimizer = SGD(self.model.parameters(), lr=0.1)

    def train(self):
        self.model.train()
        loop = tqdm(self.dataloader, total=len(self.dataloader))
        for x, y in loop:
            x, y = x.to(device), y.to(device)

            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, w: dict):
        """
        load model parameter from server
        :param w: parameter downloaded from server
        """
        self.model.load_state_dict(w)

class FedavgServer(FlServer):
    def __init__(self, model: nn.Module, test_dataloader: DataLoader, loss_fn):
        self.loss_fn = loss_fn
        self.test_dataloader = test_dataloader
        self.model = model

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, weight: dict):
        self.model.load_state_dict(weight)

    def aggregate(self, weights: list[OrderedDict]):
        """
        aggregate weights from clients and set aggregated weight
        """
        size = len(weights)
        ans = None
        for weight in weights:
            if ans is None:
                ans = weight
            else:
                for key, value in weight.items():
                    ans[key] += value
        for key in ans.keys():
            ans[key] = torch.div(ans[key], size)
        self.model.load_state_dict(ans)
        self.model.eval()

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        acc = 100 * correct
        print(f"Accuracy: {acc:>0.2f}%, Avg loss: {test_loss:>8f}")
        return acc

if __name__ == '__main__':
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    client_size = 10
    transmission_round = 10
    local_epoch_size = 5

    iid_partition = CIFAR10Partitioner(training_data.targets, client_size, balance=True, partition="iid")
    diri_partition = CIFAR10Partitioner(training_data.targets, client_size, partition="dirichlet", dir_alpha=0.3)

    iid_set = [torch.utils.data.Subset(training_data, iid_partition[i]) for i in range(client_size)]
    dirichlet_set = [torch.utils.data.Subset(training_data, diri_partition[i]) for i in range(client_size)]

    test_dataloader = DataLoader(test_data, batch_size=128)
    iid_train_dataloader = [DataLoader(iid_set[i], batch_size=64) for i in range(client_size)]
    non_iid_train_dataloader = [DataLoader(dirichlet_set[i], batch_size=64) for i in range(client_size)]

    server = FedavgServer(resnet18().to(device), test_dataloader, nn.CrossEntropyLoss())
    iid_clients = [FedavgClient(resnet18().to(device), iid_train_dataloader[i], nn.CrossEntropyLoss()) for i in range(client_size)]
    non_iid_clients = [FedavgClient(resnet18().to(device), non_iid_train_dataloader[i], nn.CrossEntropyLoss()) for i in range(client_size)]


    for t in range(transmission_round):
        print(f"Transmission {t+1}:")
        local_weights = []
        cl = 0
        for client in iid_clients:
            print(f"Client {cl}:")
            for epoch in range(local_epoch_size):
                print(f"Epoch {epoch+1}:")
                client.train()
            torch.save(client.get_params(), f"./params/client{cl}.pth")
            cl += 1
        for idx in range(client_size):
            local_weights.append(torch.load(f"./params/client{idx}.pth"))
        server.aggregate(local_weights)
        acc = server.test()
        torch.save(server.get_params(), "./params/server.pth")
        for client in iid_clients:
            client.set_params(torch.load("./params/server.pth"))

    server = FedavgServer(resnet18().to(device), test_dataloader, nn.CrossEntropyLoss())

    for t in range(transmission_round):
        print(f"Transmission {t+1}:")
        local_weights = []
        cl = 0
        for client in non_iid_clients:
            print(f"Client {cl}:")
            for epoch in range(local_epoch_size):
                print(f"Epoch {epoch+1}:")
                client.train()
            torch.save(client.get_params(), f"./params/client{cl}.pth")
            cl += 1
        for idx in range(client_size):
            local_weights.append(torch.load(f"./params/client{idx}.pth"))
        server.aggregate(local_weights)
        acc = server.test()
        torch.save(server.get_params(), "./params/server.pth")
        for client in non_iid_clients:
            client.set_params(torch.load("./params/server.pth"))