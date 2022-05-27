import matplotlib.pyplot as plt

from generateData import generate
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns

def measure_peak_height_by_ratio(spectrum:np.array, peak_index:int, stop_ratio:float) -> float:
    '''
    :spectrum: array of spectrum values:
    :peak_index: index of previously identified peak:
    :stop_ratio:difference between left and right side values that indicates being out of the peak:

    :returns: float estimated height:
    TODO: psuedocode, document, implement, test

    This will fail when peak is flat

    Pseudocode- iterative:
    ratio = 0
    distance = 0
    while ratio < stop_ratio:
        # increment look distance:
        distance += 1

        #look
        left = spectrum[peak_index - distance]
        right = spectrum[peak_index + distance]
        height = (left + right) / 2

        #calculate ratio:
        ratio = min([left, right]) / max([left, right])

    return height

    Pseudocode- recursive

    left = spectrum[peak_index - 1]
    right = spectrum[peak_index + 1]
    height = (left + right) / 2
    distance = 1

    def hill_climb(height, distance)
        if ratio > stop_ratio:
            return height
        else:

    '''
    #iterative
    ratio = 0
    distance = 0
    while ratio < stop_ratio:
        # increment look distance:
        distance += 1

        # look
        left = spectrum[peak_index - distance]
        right = spectrum[peak_index + distance]
        height = (left + right) / 2

        # calculate ratio:
        ratio = max([left, right]) / min([left, right])
        print(f'ratio = {ratio}')

    return height

def measure_peak_height_by_rate_of_change(spectrum:np.array, peak_index:int, critical_slope:float, length, plot:bool = False) -> float:
    '''

    This works! but, will need to find a good way to calibrate the length and critical slope params
    This function uses two helper functions to estimate the depth of an absorbntion line in a non-rectified spectrum.

    :spectrum: array of spectrum val;ues:
    :peak_index: int index of detected peak:
    :critical_slope: float of slope value that is considered 'out' of the trough:
    :length: length of regression line to fit:

    :Returns: float estimated height, calculated as average of two exit points:

    TODO: calibrate length, critical slope
    TODO: psuedocode, implement, document, test
    #needs to be a state machine in that the derivative will be 0 at the peak, so it must remember previous state
    1) calctulate rates of change using lm for segment of length «2 * length + 1» going right and left until critical derivative
    2) midpoint between two critical points is the height of the peak.
    '''
    sns.set_theme()
    def measure_derivative(x_segment, y_segment) -> float:
        '''
        helper function to approximate derivative using a linear model over the interval (index-length, index+length]
        '''
        #TODO: make sure it doesnt fail at the end or beginnibng of the spectrum
        lm = LinearRegression().fit(X = x_segment, y = y_segment)
        slope = lm.coef_[0]
        return slope

    def look(index, direction='right'):
        '''returns index where critical slope is found'''
        directions = {
            'right' : 1,
            'left' : -1
        }
        if direction not in directions:
            raise ValueError
        else:
            move = directions[direction]
        state = 0
        while True:
            y_segment = np.array(spectrum[index-length:index+length])
            x_segment = np.arange(0, len(y_segment)).reshape(-1, 1)
            slope = measure_derivative(x_segment, y_segment)
            if index-length < 1 or index + length > len(spectrum): #out of bounds
                return None
            if state == 1 and np.abs(slope) < critical_slope:
                return index
            elif state == 0 and np.abs(slope) > 0.9: #arbitrary low beginning slope, transition to state 1
                state = 1

            index += move

    rim_indices = [] #hold the indices of the L and R rims of the trough
    for direction in ['left', 'right']:
       rim_indices.append( look(peak_index, direction))
    if plot:
        plt.plot(spectrum)
        [plt.axvline(line, c= 'red') for line in rim_indices if line]


    heights = [spectrum[i] for i in rim_indices if i]
    height = np.mean(heights)
    if plot:
        plt.axhline(height, c='black')
   # height -= spectrum[peak_index]
    #height = - height
    if plot:
        plt.show()
    return height



if __name__ == '__main__':

    # bests = []
    # for i in range(30):
    #     data = generate(rectified=True, plot=False)
    #     slopes = np.arange(0.08, 0.3, 0.08)
    #     bandwidths = np.arange(2, 10, 1)
    #     errors = []
    #     (spectrum, peaks) = data
    #
    #     best_combo = ()
    #     best_score = -np.inf
    #
    #     for slope in slopes:
    #         errors.append([])
    #         heights = []
    #
    #         for bandwidth in bandwidths:
    #             for peak in peaks:
    #                 heights.append(measure_peak_height_by_rate_of_change(spectrum[1], peak, slope, bandwidth, plot=False))
    #             errors[-1].append(np.mean(heights))
    #
    #             if np.mean(heights) > best_score:
    #                 best_combo, best_score = (slope, bandwidth), np.mean(heights)
    #
    # print(f'best combo = {best_combo}')
    # bests.append(best_combo)
    # print('best combos:', bests)
    # slopes = [t[0] for t in bests]
    # bandwidths = [t[1] for t in bests]
    # print('average bet slope: ', np.mean(slopes))
    # print('average best bandwidth', np.mean(bandwidths))
    data = generate(rectified=True, plot=False)
    (spectrum, peaks) = data
    heights = []
    for peak in peaks:
                 heights.append(measure_peak_height_by_rate_of_change(spectrum[1], peak, 0.08, 5, plot=True))