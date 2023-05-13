import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


def run():
    """02.0 Beginning"""

    '''Creating data to view and fit'''

    # Make 1000 examples
    n_samples = 1000

    # Create circles
    X, y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)

    # Check out the features
    # print(X)
    # See the first 10 labels
    # print(y[:10])

    # Make dataframe of features and labels
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
    # print(circles.head())

    # Check out the different labels
    # print("\n", circles.label.value_counts())

    # Visualize with a plot
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()

    '''Input and output shapes'''

    # Check the shapes of our features and labels
    print(X.shape, y.shape)
    # Check how many samples we have
    print(len(X), len(y))

    # View the first example of features and labels
    print(X[0], y[0])
