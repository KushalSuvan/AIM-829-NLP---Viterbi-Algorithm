
import numpy as np
from numpy import typing as npt

class POSTagger(object):
    def __init__(self):
        self.X = None
        self.y = None

    def train(self, X: npt.NDArray[np.integer], y: npt.NDArray[np.integer]):
        assert(len(X) == len(y))
        for i in range(X.shape[0]):
            assert(len(X[i]) == len(y[i]))

        self.X = X
        self.y = y

    def predict(self, X):
        pass

import pandas as pd

df = pd.read_csv('./TRAIN - TRAIN.csv.csv')
print(df.describe())