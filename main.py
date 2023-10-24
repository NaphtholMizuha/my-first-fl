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
    client_size = 2
    transmission_round = 10
    local_epoch_size = 1

    iid_partition = CIFAR10Partitioner(training_data.targets, client_size, balance=True, partition="iid")
    # diri_partition = CIFAR10Partitioner(training_data.targets, client_size, balance=True, partition="dirichlet", dir_alpha=0.3)

    iid_set = [torch.utils.data.Subset(training_data, iid_partition[i]) for i in range(client_size)]
    # dirichlet_set = [torch.utils.data.Subset(training_data, diri_partition[i]) for i in range(client_size)]

    test_dataloader = DataLoader(test_data, batch_size=128)
    iid_train_dataloader = [DataLoader(iid_set[i], batch_size=128) for i in range(client_size)]
    # non_iid_train_dataloader = [DataLoader(dirichlet_set[i], batch_size=64) for i in range(client_size)]


    iid_clients = [FedavgClient(SimpleCnn(input_dim=16*5*5, hidden_dims=[120, 84], out_dim=10).to(device), iid_train_dataloader[i]) for i in range(client_size)]
    # non_iid_clients = [FedavgClient(SimpleCnn(input_dim=16*5*5, hidden_dims=[120, 84], out_dim=10).to(device), non_iid_train_dataloader[i]) for i in range(client_size)]
    server = FedavgServer(FedconNet().to(device), test_dataloader)
    clients = [FedconClient(FedconNet().to(device),
                            iid_train_dataloader[i], i) for i in range(client_size)]

    fedcon_acc = []
    fedcon_loss = []
    acc, loss = server.test()
    fedcon_acc.append(acc), fedcon_loss.append(loss)

    for t in range(transmission_round):
        logger.info(f"Transmission {t+1} begins")
        local_weights = []
        for idx, client in enumerate(clients):
            client.local_step(n_epoch=local_epoch_size)
            torch.save(client.get_weight(), f"./params/fedcon{idx}.pth")
        for idx in range(client_size):
            local_weights.append(torch.load(f"./params/fedcon{idx}.pth"))
        server.aggregate(local_weights)
        torch.save(server.get_weight(), "./params/fedcon-server.pth")
        for client in clients:
            client.set_weight(torch.load("./params/fedcon-server.pth"))
        acc, loss = server.test()
        logger.warning(f"Transmission {t+1} ends: acc={acc}, loss={loss}")
        fedcon_acc.append(acc), fedcon_loss.append(loss)

    server = FedavgServer(SimpleCnn(input_dim=16 * 5 * 5, hidden_dims=[120, 84], out_dim=10).to(device),
                          test_dataloader)
    acc, loss = server.test()
    fedavg_acc = []
    fedavg_loss = []
    fedavg_acc.append(acc), fedavg_loss.append(loss)

    for t in range(transmission_round):
        logger.info(f"Transmission {t+1} begins")
        local_weights = []
        cl = 0
        for client in iid_clients:
            logger.info(f"Client {client} begins")
            for epoch in range(local_epoch_size):
                loss = client.train()
                logger.debug(f"Epoch {epoch + 1}: loss={loss:>.03}")
            torch.save(client.get_weight(), f"./params/fedavg{cl}.pth")
            cl += 1
        for idx in range(client_size):
            local_weights.append(torch.load(f"./params/fedavg{idx}.pth"))
        server.aggregate(local_weights)
        acc, loss = server.test()
        torch.save(server.get_weight(), "./params/fedavg-server.pth")
        for client in iid_clients:
            client.set_weight(torch.load("./params/fedavg-server.pth"))
        logger.warning(f"Transmission {t + 1} ends: acc={acc}, loss={loss}")
        fedavg_acc.append(acc), fedavg_loss.append(loss)

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

