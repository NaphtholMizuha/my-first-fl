from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from fedlab.utils.dataset import CIFAR10Partitioner
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from frameworks import *
from models import SimpleCnn, FedconNet
import matplotlib.pyplot as plt

device = 'cuda:0'

def train_mcflcs(server, clients, t_round, n_epoch):
    acc, loss = [], []
    acc_, loss_ = server.test()
    acc.append(acc_), loss.append(loss_)
    for t in range(t_round):
        logger.info(f"Transmission {t+1} begins")
        local_weights = []
        for idx, client in enumerate(clients):
            w_i_t = client.local_step(n_epoch)
            local_weights.append(w_i_t)
        server.aggregate(local_weights)
        for client in fedcon_clients:
            client.set_weight(deepcopy(server.get_weight()))
        acc_, loss_ = server.test()
        logger.info(f"Transmission {t+1} ends: acc={acc}, loss={loss}")
        acc.append(acc_), loss.append(loss_)
    return acc, loss

def train_fedavg(server, clients, t_round, n_epoch):
    for t in range(t_round):
        acc, loss = [], []
        acc_, loss_ = server.test()
        acc.append(acc_), loss.append(loss_)
        logger.info(f"Transmission {t+1} begins")
        local_weights = []
        for client in clients:
            w_i = client.local_step(n_epoch)
            local_weights.append(deepcopy(w_i))
        server.aggregate(local_weights)
        acc, loss = server.test()
        for client in clients:
            client.set_weight(deepcopy(server.get_weight()))
        logger.info(f"Transmission {t + 1} ends: acc={acc}, loss={loss}")
        acc.append(acc), loss.append(loss)

def plot_contrast(*lists):

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
    transmission_round = 25
    local_epoch_size = 5

    partition = CIFAR10Partitioner(training_data.targets, client_size, balance=True, partition="iid")
    dataset = [torch.utils.data.Subset(training_data, partition[i]) for i in range(client_size)]
    test_dataloader = DataLoader(test_data, batch_size=64)
    train_dataloader = [DataLoader(dataset[i], batch_size=64, shuffle=True) for i in range(client_size)]

    fedavg_clients = [FedavgClient(resnet18().to(device), train_dataloader[i], test_dataloader, i) for i in range(client_size)]
    server = FedavgServer(FedconNet(base="resnet18").to(device), test_dataloader)
    fedcon_clients = [FedconClient(FedconNet(base="resnet18").to(device),
                                   train_dataloader[i], test_dataloader, i) for i in range(client_size)]




    plt.plot(fedavg_acc, color="blue")
    plt.plot(fedcon_acc, color="red")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Accuracy")
    plt.show()
    plt.plot(fedavg_loss, color="blue")
    plt.plot(fedcon_loss, color="red")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Loss")
    plt.show()

