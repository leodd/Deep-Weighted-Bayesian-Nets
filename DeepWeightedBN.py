import numpy as np
import itertools
import torch
import torch.nn as nn
from Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction

# from NeuralNetPotential import NeuralNetFunction, ReLU


class DeepWeightedBN:
    def __init__(self, domains, nn_setting):
        self.domains = domains

        self.dim_x = len(domains) - 1
        self.BN_idx_list = list(itertools.chain.from_iterable(
            itertools.combinations(np.arange(self.dim_x), r) for r in range(1, self.dim_x)
        ))

        self.naive_BN_units = list()
        self.naive_BN_prior = None

        nn_setting_list = [nn.Linear(self.dim_x, nn_setting[0]), nn.ReLU()] + list(itertools.chain.from_iterable(
            (nn.Linear(nn_setting[i], nn_setting[i + 1]), nn.ReLU()) for i in range(len(nn_setting) - 1)
        )) + [nn.Linear(nn_setting[-1], len(self.BN_idx_list))]

        self.nn = nn.Sequential(*nn_setting_list)

        # self.nn = NeuralNetFunction(
        #     (self.dim_x, nn_setting[0], ReLU()),
        #     *[(nn_setting[i], nn_setting[i + 1], ReLU()) for i in range(len(nn_setting) - 1)],
        #     (nn_setting[-1], len(self.BN_idx_list), None),
        # )

    def train_BN(self, data):
        table = np.zeros(shape=len(self.domains[-1].values))
        idx, count = np.unique(data[:, -1].astype(int), return_counts=True, axis=0)
        table[tuple([idx])] = count
        table /= np.sum(table)

        self.naive_BN_prior = TableFunction(table)

        self.naive_BN_units.clear()

        for i in range(self.dim_x):  # Learn the conditional probability p(x|y)
            w_table = np.ones(shape=len(self.domains[-1].values))
            dis_table = np.zeros(shape=w_table.shape, dtype=int)

            dis = [GaussianFunction(np.zeros(1), np.eye(1))]

            for y in idx:
                data_xi = data[data[:, -1] == y, i]

                if len(data_xi) <= 1:
                    continue

                mu = [np.mean(data_xi)]
                sig = [[np.var(data_xi)]]

                dis_table[y] = len(dis)
                dis.append(GaussianFunction(mu, sig))

            self.naive_BN_units.append(CategoricalGaussianFunction(w_table, dis_table, dis, [self.domains[i], self.domains[-1]]))

    def infer_BN(self, x):
        y_naive_BN_units = list()
        for i in range(self.dim_x):
            temp = np.empty([len(x), len(self.domains[-1].values)])
            for y in range(len(self.domains[-1].values)):
                temp[:, y] = self.naive_BN_units[i].batch_call(np.hstack([
                    x[:, [i]],
                    np.ones([len(x), 1]) * y
                ]))
            y_naive_BN_units.append(temp)

        y_naive_BNs = np.empty([len(x), len(self.BN_idx_list), len(self.domains[-1].values)])
        for i, features_idx in enumerate(self.BN_idx_list):
            temp = self.naive_BN_prior.table
            for j in features_idx:
                temp = y_naive_BN_units[j] * temp
            y_naive_BNs[:, i, :] = temp

        return y_naive_BNs

    def train(self, data, max_iter=1000, lr=0.001, regular=0.001):
        self.train_BN(data)

        y_naive_BNs = torch.from_numpy(self.infer_BN(data)).float()

        data_x = torch.from_numpy(data[:, :-1]).float()
        data_y = torch.from_numpy(data[:, -1]).long()

        optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr, weight_decay=regular)
        loss = nn.CrossEntropyLoss()

        for t in range(max_iter):
            optimizer.zero_grad()
            w = self.nn(data_x)
            w = w.unsqueeze(2)

            y = torch.sum(w * y_naive_BNs, dim=1)
            output = loss(y, data_y)

            print(output)

            output.backward()
            optimizer.step()

    def infer(self, x):
        y_naive_BNs = torch.from_numpy(self.infer_BN(x)).float()

        x = torch.from_numpy(x).float()

        with torch.no_grad():
            w = self.nn(x)
            w = w.unsqueeze(2)

            y = torch.sum(w * y_naive_BNs, dim=1)
            _, y = torch.max(y, dim=1)

        return y.numpy()
