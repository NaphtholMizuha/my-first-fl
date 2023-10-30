from copy import deepcopy

import pandas
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from frameworks import *
from models import SimpleCnn, FedconNet, ThreeLayers
import matplotlib
import pandas as pd

device = 'cuda'


def train_net(server, clients, t_round, n_epoch):
    acc, loss = [], []
    for t in range(t_round):
        logger.info(f"Transmission {t + 1} begins")
        local_weights = []
        for client in clients:
            w_i = client.local_step(n_epoch)
            local_weights.append(w_i)
        server.aggregate(local_weights)
        acc_, loss_ = server.test()
        for client in clients:
            client.set_weight(deepcopy(server.get_weight()))
        logger.info(f"Transmission {t + 1} ends: acc={acc_}, loss={loss_}")
        acc.append(acc_), loss.append(loss_)
    return acc, loss


if __name__ == '__main__':
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    client_size = 10
    transmission_round = 25
    local_epoch_size = 5
    matplotlib.use("Agg")
    partition = utils.dataset_partition(training_data.targets, 0.5, client_size, "dirichlet")
    freq = {i: len(partition[i]) / len(training_data) for i in range(client_size)}
    utils.show_distribution(training_data, partition, client_size, 10)
    dataset = [torch.utils.data.Subset(training_data, partition[i]) for i in range(client_size)]
    test_dataloader = DataLoader(test_data, batch_size=64)
    train_dataloader = [DataLoader(dataset[i], batch_size=64, shuffle=True) for i in range(client_size)]
    net_sample = FedconNet(base="fc").to(device)
    pack_table = utils.gen_pack_table(net_sample.state_dict())
    McflcsClient.table = pack_table
    McflcsServer.table = pack_table

    mcflcs_server = McflcsServer(net_sample.to(device), test_dataloader, freq)
    mcflcs_clinets = [McflcsClient(FedconNet(base="fc").to(device),
                                   train_dataloader[i], test_dataloader, i) for i in range(client_size)]
    fedcon_server = FedavgServer(FedconNet(base="fc").to(device), test_dataloader, freq)
    fedcon_clients = [FedconClient(FedconNet(base="fc").to(device), train_dataloader[i], test_dataloader, i)
                      for i in range(client_size)]
    fedavg_server = FedavgServer(ThreeLayers().to(device), test_dataloader, freq)
    fedavg_clients = [FedavgClient(ThreeLayers().to(device), train_dataloader[i], test_dataloader, i)
                        for i in range(client_size)]

    fedcon_acc, fedcon_loss = train_net(fedcon_server, fedcon_clients, transmission_round, local_epoch_size)
    fedavg_acc, fedavg_loss = train_net(fedavg_server, fedavg_clients, transmission_round, local_epoch_size)
    mcflcs_acc, mcflcs_loss = train_net(mcflcs_server, mcflcs_clinets, transmission_round, local_epoch_size)

    acc_df = pandas.DataFrame([fedavg_acc, mcflcs_acc, fedcon_acc])
    loss_df = pandas.DataFrame([fedavg_loss, mcflcs_loss, fedcon_loss])
    acc_df.to_csv("./results/acc.csv")
    acc_df.to_csv("./results/loss.csv")

    utils.plot_contrast("Accuracy-NonIID", [fedavg_acc, mcflcs_acc, fedcon_acc], ["FedAvg", "Mcfl-CS", "FedCon"])
    utils.plot_contrast("Loss-NonIID", [fedavg_loss, mcflcs_loss, fedcon_loss], ["FedAvg", "Mcfl-CS", "FedCon"])

    # privacy_budgets = [50, 100, 200]
    # acc_li, loss_li = [], []
    #
    # for i in range(3):
    #     server = McflcsServer(FedconNet(base="cnn").to(device), test_dataloader, freq)
    #     clinets_ = [McflcsClient(FedconNet(base="cnn").to(device), train_dataloader[j], test_dataloader, j,
    #                             n_comm=transmission_round, n_client=client_size, privacy_budget=privacy_budgets[i])
    #                             for j in range(client_size)]
    #     acc, loss = train_net(server, clinets_, transmission_round, local_epoch_size)
    #     acc_li.append(acc);
    #     loss_li.append(loss)
    #
    # acc_df = pd.DataFrame(acc_li); loss_df = pd.DataFrame(loss_li)
    # acc_df.to_csv("./results/acc.csv")
    # loss_df.to_csv("./results/loss.csv")
    #
    # utils.plot_contrast("Accuracy over privacy_budgets", acc_li, [
    #     f"$\\epsilon = {privacy_budgets[i]}$" for i in range(3)
    # ])
    #
    # utils.plot_contrast("Loss over privacy_budgets", loss_li, [
    #     f"$\\epsilon = {privacy_budgets[i]}$" for i in range(3)
    # ])