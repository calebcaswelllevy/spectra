from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from generateData import generate
    data, peaks = generate(rectified = False)


    xx, yy, zz = kde2D(data[0], data[1], 60)

    plt.pcolormesh(xx, yy, zz)
    plt.scatter(data[0], data[1], s=2, facecolor='white')
    plt.show()
