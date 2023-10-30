import copy
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from loguru import logger

import utils
from models import FedconNet
from utils import dct_mat
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
    def get_weight(self):
        pass

    @abstractmethod
    def set_weight(self, w):
        pass


class FlClient(ABC):
    """
    abstract base class representing client in FL
    """

    def train(self):
        pass

    @abstractmethod
    def get_weight(self):
        pass

    @abstractmethod
    def set_weight(self, w):
        pass


class FedavgClient(FlClient):
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, idx):
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.idx = idx

    def local_step(self, n_epoch):
        """
        :param n_epoch: amount of local epochs
        :param temperature: hyperparameter in contrastive learning
        :param mu: hyperparameter for the impact of contrastive learning
        :return: local weight after training
        """
        # logger.debug(f"Client {self.idx} begins")
        for epoch in range(n_epoch):
            loss = self.train()
            # logger.debug(f"Epoch {epoch + 1}: loss={loss:.5}")
        # acc, loss = self.test()
        # logger.debug(f"Client {self.idx} ends: acc={acc}, loss={loss}")
        return self.get_weight()

    def test(self):
        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to("cuda"), y.to("cuda")
                if isinstance(self.model, FedconNet):
                    _, pred = self.model(x)
                else:
                    pred = self.model(x)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return correct, test_loss

    def train(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to("cuda"), y.to("cuda")

            pred = self.model(x)
            loss = self.criterion(pred, y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / len(self.test_loader)

    def get_weight(self):
        return self.model.state_dict()

    def set_weight(self, w: dict):
        """
        load model parameter from server
        :param w: parameter downloaded from server
        """
        self.model.load_state_dict(w)

class FedconClient(FlClient):

    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, idx: int):
        super().__init__()
        self.test_loader = test_loader
        self.model = model.to('cuda')
        self.model_prev = copy.deepcopy(model).to('cuda')
        self.model_glob = copy.deepcopy(model).to('cuda')
        self.idx = idx
        self.train_loader = train_loader
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss().to("cuda")


    def get_weight(self):
        return self.model.state_dict()

    def set_weight(self, weight: dict):
        self.model.load_state_dict(weight)
        self.model_glob.load_state_dict(weight)
        self.model.eval()
        self.model_glob.eval()

    def local_step(self, n_epoch, temperature=0.5, mu=5):
        """
        :param n_epoch: amount of local epochs
        :param temperature: hyperparameter in contrastive learning
        :param mu: hyperparameter for the impact of contrastive learning
        :return: local weight after training
        """
        # logger.debug(f"Client {self.idx} begins")
        temp = copy.deepcopy(self.model.state_dict())
        for epoch in range(n_epoch):
            loss, loss1, loss2 = self.train(temperature, mu)
            # logger.debug(f"Epoch {epoch + 1}: loss={loss:.5}, loss_sup={loss1:.5}, loss_con={loss2:.5}")
        # acc, loss = self.test()
        # logger.debug(f"Client {self.idx} ends: acc={acc}, loss={loss}")
        self.model_prev.load_state_dict(temp)
        self.model_prev.eval()
        return self.get_weight()

    def train(self, temperature, mu):
        n_batch = len(self.train_loader)
        self.model.train()
        total_loss, total_loss1, total_loss2 = 0.0, 0.0, 0.0
        cos = torch.nn.CosineSimilarity(dim=-1)

        for x, y in self.train_loader:
            x, y = x.to("cuda"), y.to("cuda")
            self.optimizer.zero_grad()
            y = y.long()
            z, y_pred = self.model(x)
            z_glob, _ = self.model_glob(x)
            z_prev, _ = self.model_prev(x)

            pos = cos(z, z_glob).reshape(-1, 1)
            neg = cos(z, z_prev).reshape(-1, 1)
            logits = torch.cat((pos, neg), dim=1) / temperature
            labels = torch.zeros(x.size(0)).to('cuda').long()
            loss1 = self.criterion(y_pred, y)
            loss2 = mu * self.criterion(logits, labels)
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

        return total_loss / n_batch, total_loss1 / n_batch, total_loss2 / n_batch

    def test(self):
        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to("cuda"), y.to("cuda")
                if isinstance(self.model, FedconNet):
                    _, pred = self.model(x)
                else:
                    pred = self.model(x)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return correct, test_loss

class McflcsClient(FedconClient):
    table = {}

    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, idx: int, n_compress=100,
                 n_chunk=200, threshold=2, privacy_budget=1, n_comm=25, n_client=10):
        super().__init__(model, train_loader, test_loader, idx)
        self.n_client = n_client
        self.n_comm = n_comm
        self.n_chunk = n_chunk
        self.n_compress = int(n_compress)
        self.threshold = threshold
        self.epsilon = privacy_budget

    def get_weight(self):
        w, w_prev = self.model.state_dict(), self.model_prev.state_dict()
        w = utils.pack(w, self.n_chunk, self.table)
        w_prev = utils.pack(w_prev, self.n_chunk, self.table)
        w = w - w_prev
        w = utils.compress(w, self.n_chunk, self.n_compress)
        w = utils.differential_privacy(w, self.n_client, self.n_comm, self.threshold, self.epsilon)
        return w

class FedavgServer(FlServer):
    def __init__(self, model: nn.Module, test_dataloader: DataLoader, freq):
        self.freq = freq
        self.criterion = nn.CrossEntropyLoss()
        self.test_dataloader = test_dataloader
        self.model = model

    def get_weight(self):
        return self.model.state_dict()

    def set_weight(self, weight: dict):
        self.model.load_state_dict(weight)
        self.model.eval()

    def aggregate(self, weights: list):
        """
        aggregate weights from clients and set aggregated weight
        """
        size = len(weights)
        ans = {}
        for idx, weight in enumerate(weights):
            if idx == 0:
                for key in weight.keys():
                    ans[key] = weight[key] * self.freq[idx]
            else:
                for key in weight.keys():
                    ans[key] += weight[key] * self.freq[idx]
        self.set_weight(ans)
        self.model.eval()

    def test(self):
        self.model.eval()
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x, y = x.to("cuda"), y.to("cuda")
                if isinstance(self.model, FedconNet):
                    _, pred = self.model(x)
                else:
                    pred = self.model(x)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return correct, test_loss

class McflcsServer(FedavgServer):
    table = {}
    def __init__(self, model: nn.Module, test_dataloader: DataLoader, freq, n_compress=100,
                 n_chunk=200, rho=0.1, eta=0.1, n_clients=10):
        super().__init__(model, test_dataloader, freq)
        self.n_clients = n_clients
        self.eta = eta
        self.n_chunk = n_chunk
        self.n_compress = int(n_compress)
        self.momentum = None
        self.residual = None
        self.rho = rho
        self.w_prev = utils.pack(self.model.state_dict(), n_chunk, self.table)

    # def set_weight(self, w_compressed):
    #     w_in_chunks = utils.reconstruct(w_compressed, self.n_chunk, self.n_compress)
    #     w_dict = utils.unpack(w_in_chunks, self.tableff)
    #     self.model.load_state_dict()

    def aggregate(self, weights: list):
        y = torch.zeros(weights[0].size()).to('cuda')
        for i, weight in enumerate(weights):
            y += self.freq[i] * weight
        if self.momentum is None:
            self.momentum = torch.zeros(weights[0].shape).to('cuda')
        if self.residual is None:
            self.residual = torch.zeros(weights[0].shape).to('cuda')
        self.momentum = y + self.rho * self.momentum
        self.residual += self.eta * self.momentum
        s = utils.reconstruct(self.residual, self.n_chunk, self.n_compress)
        self.residual -= utils.compress(s, self.n_chunk, self.n_compress)
        w_pack = self.w_prev + s
        w = utils.unpack(w_pack, self.table)
        self.w_prev = w_pack
        self.set_weight(w)
