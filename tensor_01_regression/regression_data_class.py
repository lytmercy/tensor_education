import numpy as np


class RegressionData:

    def __init__(self):
        self.X = np.arange(-100, 100, 4)
        self.y = np.arange(-90, 110, 4)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train_test_init(self):
        self.X_train = self.X[:40]
        self.y_train = self.y[:40]

        self.X_test = self.X[40:]
        self.y_test = self.y[40:]
