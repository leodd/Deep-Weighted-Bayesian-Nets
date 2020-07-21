from DeepWeightedBN import DeepWeightedBN
from Graph import Domain
import numpy as np
import pandas as pd


data = pd.read_csv('data/iris.data')
data = data.to_numpy()

d_x = Domain([-np.Inf, np.Inf], continuous=True)
d_y = Domain(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], continuous=False)
domains = [d_x, d_x, d_x, d_x, d_y]

data[:, -1] = d_y.value_to_idx(data[:, -1])

dwbn = DeepWeightedBN(domains, nn_setting=[20, 20])
dwbn.train(data)
