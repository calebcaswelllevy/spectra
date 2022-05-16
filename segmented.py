from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
from generateData import derectify
from functionTimer import measure_time



@measure_time
def KDE_rectifier(data, alpha, segment_width ):
    """

    The idea here would be to fit a kde, remove the biggest outliers, fit a kde, remove outliers etc until some
    stopping condition. The result would be the most 'average points'

     kernel, bandwidth <- optimize kde params
    data = self.data.copy()
    While outliers:
        probabilities = [1 for i in data - 2 * segment_width]
        for i between segment_width and n - segment_width + 1:
            fit kernel to data centered on data[i], ranging from data[i-segment_width] to data[i+segment_width+1]
            p = evaluate probability of point
            probabilities[i] = p
        outliers, tau = self.findOutliers(p, alpha)
        #drop outliers
        data = data[data[0] not in outliers]

    #fit a spline to the leftover points
    #predict missing values using the spline
    #subtract the spline from the spectrum


    """
    # copy the data
    data = data.copy()
    # parameter vals to evaluate
    h_vals = np.arange(0.05, 1, 0.3)
    # Kernel types to evaluate:
    kernels = ['epanechnikov']

    # Optimize h (smoothing parameter) and kernel type:
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': h_vals,
                         'kernel': kernels
                         })
    grid.fit(np.array(data[1]).reshape(-1, 1) )
    params = grid.best_params_
    bandwidth = params['bandwidth']

    # outer loop:
    outliers = True

    while outliers:
        n = len(data[0])
        probabilities = [0 for i in range(len(data[0]))]
        for i in range(segment_width, n - segment_width + 1):
            sub_data = data[i]
            kernel = KernelDensity(bandwidth = bandwidth, kernel="epanechnikov")


def get_weights(n:int) -> tuple:
    '''
    TODO: test
    method to apply weights for a weighted average based on a logistic derivative shape
    '''

    weights = np.array([1 for i in range(len(data))])
    weights = np.array(derectify(weights))
    return weights

@measure_time
def LM_rectifier(data, segment_width, n_reps ):
    '''
    TODO: implement
    1) partition the data into segment_width partitions
    for i in N:
    2) fit a linear model to each partition
    3) predict a line for each segment
    4) subtract prediction from portion of spectrum
    5)
    '''

@measure_time
def LM_model_average_rectifier(data, bandwidth, n_reps):
    '''
    TODO: implement
    n = range(len(data)
    1) for i in n_reps
        model_estimates = [0 for i in n]
        counts = [0 for i in n]
        weights = get_weights(n = bandwidth * 2 + 1)
         for index, wavelength in data[bandwidth : len(data) - bandwidth + 1]:
            #can optimize this list creation using a queue:
            X = [index-bandwidth:index+bandwidth+1]
            subset_centered_on_wavelength = data[index-bandwidth:index+bandwidth+1]
            model <- fit linear model to subset_centered_on_wavelength
            local_predictions = model.predict()
            weighted_local_predictions = local_predictions * weights
            for index, prediction in zip(X, local_predictions):


    '''