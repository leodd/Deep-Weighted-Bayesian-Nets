import numpy as np
import itertools
from NeuralNetPotential import NeuralNetFunction, ReLU
from Potentials import GaussianFunction, TableFunction, CategoricalGaussianFunction


class DeepWeightedBN:
    def __init__(self, domains, nn_setting):
        self.domains = domains

        self.dim_x = len(domains) - 1
        self.BN_idx_list = list(itertools.chain.from_iterable(
            itertools.combinations(np.arange(self.dim_x), r) for r in range(1, self.dim_x)
        ))

        self.naive_BN_units = list()
        self.naive_BN_prior = None
        self.nn = NeuralNetFunction(
            (self.dim_x, nn_setting[0], ReLU()),
            *[(nn_setting[i], nn_setting[i + 1], ReLU()) for i in range(len(nn_setting) - 1)],
            (nn_setting[-1], len(self.BN_idx_list), None),
        )

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

    def train(self, data):
        self.train_BN(data)

        y_naive_BN_units = list()
        for i in range(self.dim_x):
            temp = np.empty([len(data), len(self.domains[-1].values)])

            for y in range(len(self.domains[-1].values)):
                temp[:, y] = self.naive_BN_units[i].batch_call(np.hstack([
                    data[:, [i]],
                    np.ones([len(data), 1]) * y
                ]))

            y_naive_BN_units.append(temp)

        print(y_naive_BN_units)
