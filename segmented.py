from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import numpy as np
from generateData import generate
import generateData
from generateData import derectify
from functionTimer import measure_time
import matplotlib.pyplot as plt
import seaborn as sns



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


def get_weights(n:int, **kwargs) -> np.array:
    '''
    method to apply weights for a weighted average based on a logistic derivative shape

    :n: integer value of weights to calculate:

    :Returns: numpy array of weight values:
    '''

    weights = np.arange(-n, n+1)
    weights = np.array(derectify(weights, **kwargs))
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
def LM_model_average_rectifier(data:np.array, bandwidth:int, n_reps:int = 3, weighted:bool = True, plot:bool = False):
    '''
    This function iteratively fits a series of linear models to the data, each centered on a point and extending bandwidth length
    in both directions from that point. It then predicts the value for the function based on those models, and takes the average
    value for each point. If weighted is set to true, the value will be discounted by a hump-shaped function so central values
    are weighted higher. It then subtracts this average value from the original curve.

    :data: Numpy array of data to be rectified:
    :bandwidth: integer number of values to left and right to consider in lm fit:
    :n_reps:  integer number of times to repeat the process:
    :weighted: boolean value reflecting whether to perform weighted average calculations:
    :plot: bool whether to create a diagnostic plot of the process:

    :Returns: numpy array of rectified data:

    Based off this psuedocode:
    n = range(len(data)
    for i in n_reps
        predictions = [0 for i in n]

        total_weights = [0 for i in n]
        weights = get_weights(n = bandwidth * 2 + 1)
         for index, wavelength in data[bandwidth : len(data) - bandwidth + 1]:
            #can optimize this list creation using a queue:
            X = [index-bandwidth:index+bandwidth+1]
            subset_centered_on_wavelength = data[index-bandwidth:index+bandwidth+1]
            model <- fit linear model to subset_centered_on_wavelength
            local_predictions = model.predict()
            weighted_local_predictions = local_predictions * weights
            for index, prediction, weight in zip(X, weighted_local_predictions, weights):
                #add weighted prediction to predictions at index x
                predictions[x] += prediction
                #add weight to total_weights at index x
                weights[x] += weight

        predictions /= weights


        data -= predictions
    '''
    data2 = data.copy()

    a = 0
    n = len(data2)

    #to test performance:
    average_distance_from_0 = np.mean(data2)


    averages = [average_distance_from_0]

    window_size = bandwidth * 2 + 1
    stop_point = n - 2*bandwidth
    p=plt.figure(figsize=(8,10))

    for i in range(n_reps):
        #p
        #
        predictions = [0 for i in range(n)]

        total_weights = [0 for i in range(n)]
        if weighted:
            spread_parameter = bandwidth/4
            weights = get_weights(n = bandwidth, stretch_x = spread_parameter)
        else:
            weights = [1 for i in range(window_size)]

        for index, wavelength in enumerate(data2[bandwidth : stop_point]):

            slice_start, slice_stop = index + bandwidth, index + bandwidth + window_size

            #TODO next two lines can be optimized using a queue data structure:
            #
            X = np.arange(start=slice_start, stop = slice_stop)

            y = data2[slice_start : slice_stop]
            y = interpolate_nan_and_negative(y)


            if slice_stop >= len(data2):
                a+=1
                #print(f'a problem {a} at index {index}')
            #fit a linear model to subset of data:


            lm = LinearRegression().fit(X.reshape(-1, 1) , y)
            local_predictions = lm.predict(X.reshape(-1, 1))
            weighted_local_predictions = local_predictions * weights

            #TODO can optimize this with np array vectorized addition


            for index, prediction, weight in zip(X, weighted_local_predictions, weights):
                #add weighted prediciton to predictions at index:
                predictions[index]  += prediction
                #add weight to total_weights at index:
                total_weights[index] += weight

        predictions = np.array(predictions) / np.array(total_weights)
        if plot:
            plt.plot(predictions, c="blue", alpha=0.7, label='prediction')
            plt.plot(data, label='data', c='red', alpha=0.7)
        data2 = data2 - predictions

        #look at performance:
        average_distance_from_0 = np.nanmean(data2)
        averages.append(average_distance_from_0)



        if plot:
            #sns.set_theme(palette=sns.diverging_palette(150, 275, s=80, l=55, n=n_reps+1))
            sns.set_theme(palette = sns.color_palette("Spectral", n_colors=n_reps+1))
            plt.plot(data2, alpha = 0.6, label=f'{i+1}')


           # plt.plot(data, alpha = 0.4)

            plt.axhline(0, c='red', alpha = 0.6)

        #plt.title(f'rep {i}')
    if plot:
        plt.tight_layout()
        plt.legend(title = 'iteration', loc='best')
        plt.title(f'{n_reps} iterations with bandwidth of {bandwidth}')
        plt.show()
        p.savefig(f'{n_reps} iterations with bandwidth of {bandwidth} weighted.jpg')

    #plt.plot(averages)

    #print(averages)
    #plt.show()
    return data2


def interpolate_nan_and_negative(data: np.array) -> np.array:
    data2 = np.array(data)
    for index, number in enumerate(data2):
        if number < 0:
            data2[index] = 0
    return data2

def polynomial_spline_rectifier():
    '''
    TODO: psuedocode, implement, document
    Should fit a polynomial spline to spectrum and subtract it out.
    '''

    pass

if __name__ == '__main__':
    data = generate(rectified=False)

    bandwidths = np.arange(30, 151, 30)


    for bandwidth in bandwidths:
        data2 = np.array(data[0][1])
        data2 = LM_model_average_rectifier(data2, bandwidth, 20, weighted = True)


    # data = generate(rectified = False)
    #
    # data2 = np.array(data[0][1])
    # data2 = LM_model_average_rectifier(data2, 100, 9, weighted = False)
    #


