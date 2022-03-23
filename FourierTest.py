import parseData, numpy as np, pandas as pd, matplotlib.pyplot as plt


def DFT(x:pd.Series):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)

    return x


