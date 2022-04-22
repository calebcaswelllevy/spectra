import numpy as np, matplotlib.pyplot as plt, pandas as pd


def generate( n:int=1000, nPeaks:int = 10, noise:float = 7, peakwidthsVariability:float = 1, peakHeightVariability:float = 20, peak_set:list = [], rectified = True) -> list:
    """
    Generates an artificial spectrum. Random noise is modeled as a normal distribution centered on zero with a SD controlled by the noise parameter. The peaks are randomly generated and modeled by parabolas. Their height and width are random drawn from a uniform distribution, and are modifiable using the peakWidthsvariability and peakheightvariability parameters.

    n: the size of the spectrum to return
    nPeaks: the number of peaks in the spectrum
    noise: the sigma squared parameter of the normal distribution that models the noise of the spectrum. bigger means noisier.
    peakwidthsVariability: how much random variation in the width of the peaks?
    peakHeightVariability: how much random variation in the height of the peaks?

    Returns: a list with two elements. 1) the spectrum as a pandas DataFrame, and 2) the indices of the centers of the simulated peaks. 
    """
    if len(peak_set) < 1:
        #generate peak set with random draws
        peak_set = np.random.randint(low=20, high=n-20,size=nPeaks)
    minimumPeakWidth = 10-peakwidthsVariability
    maximumPeakWidth = 10+peakwidthsVariability
    peakwidths = np.random.randint(low=minimumPeakWidth, high=maximumPeakWidth, size=nPeaks)
    peakHeights = [np.random.rand() * peakHeightVariability + 90 for i in range(nPeaks)]

    #an initial string of random numbers subtracted from 0:

    data = np.array([number if number<=0 else -number for number in np.random.normal(loc = 0, scale = noise, size =n)])
    if not rectified: # start with hump shaped data with noise:
        hump = derectify(spectrum = np.linspace(-10, 10, num = n), stretch_y = 3*n)
        plt.subplots(3,1)
        plt.subplot(311)
        plt.plot(hump)
        data = np.array([intensity - error if error > 0 else intensity + error for intensity, error in zip(hump, data)])
        plt.subplot(312)
        plt.plot(data)

        #Make the peaks and add them to the data:
    for center, width, height in zip(peak_set, peakwidths, peakHeights):
        peak = make_peak(width=width, height=height)
        for index1, index2 in enumerate(range(center-width, center+width+1)):
            data[index2] -= peak[index1]

    data = pd.DataFrame(zip([i for i in range(len(data))], data))

    plt.subplot(313)
    plt.plot(data[1])
    [plt.axvline(peak, c='red', alpha = 0.6) for peak in peak_set]
    plt.show()
    # if rectified:
    #     data = pd.DataFrame(zip([i for i in range(len(data))], data))
    #
    # else: #Not rectified
    #     data = pd.DataFrame(zip([i for i in range(len(data))], derectify(data)))
    #s(x)(1-s(x)) | s(x) = e**x / e**x + 1 
    # s()' = (e**x / (e**x + 1)) / ((e**x / (e**x + 1)) + 1)
    
    return [data, peak_set]
    

def make_peak(width:int, height:float=100) -> list:
    """
    This is an accessory function that generates a parabolic peak of width and height given in argments. 

    width: integer width of the peak
    height: float height of peak

    Returns: list of values of the peak
    """
   
    rightSide = [-x**2 for x in range(width+1)]
    leftSide = [x for x in rightSide[:0:-1]]

    peak = leftSide+rightSide
    scale = -height/leftSide[0]
    peak = [(x*scale)+height for x in peak]
    
    return peak

def generate_peakset(number_of_peaks:int, min:float, max:float) -> np.array:
    """
    An accessory function that generates a list of wavelengths between min and max 

    number_of_peaks: how many random peaks to return
    min: lowest wavelength
    max: highest wavelength
    """

    return np.random.randint(low = min, high = max, size = number_of_peaks)

def make_group(n = 10, nPeaks = 20, spectrum_size = 2000):
    """
    """
    peak_locations = generate_peakset(number_of_peaks = nPeaks, min=20, max=spectrum_size-21)
    spectra = []
    for i in range(n):
        spectra.append(generate(n=spectrum_size, nPeaks=nPeaks, peak_set=peak_locations))
    return spectra

def derectify(spectrum, slide_x:float=0, slide_y:float=1, stretch_x:float=2.5, stretch_y:float=5, calculate_values:bool = False):
    """TO DO"""
    if calculate_values:
        n = len(spectrum)
        stretch_y = n ** 0.5
        stretch_x = 2.5


    def stretch_and_slide(func):
        def inner(x_axis):
            x = [(x / stretch_x) - slide_x for x in x_axis]
            Y = func(x)
            return [(y*stretch_y)+slide_y for y in Y]
        return inner

    @stretch_and_slide
    def derivative_of_logistic(x_axis:np.float64)->np.float64:
        """
        Function that calculates the derivative of the logistic. this is a hump-shaped function that can be used to simulate a non-rectified spectrum
        """

        e = np.e

        y = [((e**x / (e**x + 1)) * (1- (e**x / (e**x + 1)))) for x in x_axis]
        print(f'y = {y}')

        return y


    return derivative_of_logistic(spectrum)
# 1) what fraction of n to slide x NONE
# 2) what fraction of n to stretch x 2-3
# 3) what fraction of n to stretch y: root n
if __name__ == '__main__':
    X = [i for i in np.linspace(start=-10, stop=10, num=500)]
    Y = derectify(X)
    #[plt.axvline(x=peak, c="red") for peak in data[1]]
    # plt.subplots(6,1)
    # plt.subplot(611)
    # plt.plot(Y)
    # plt.title("no changes")
    #
    # plt.subplot(612)
    # Y = derectify(X, stretch_x=2)
    # plt.plot( Y)
    # plt.title("tretch x 2")
    #
    # #slide x doesnt work
    # plt.subplot(613)
    # Y = derectify(X, stretch_x=3)
    # plt.plot( Y)
    # plt.title("stretch x 3")
    #
    # plt.subplot(614)
    # Y = derectify(X, stretch_x=2, stretch_y=50)
    # plt.plot(Y)
    # plt.title("stretch x 2, stretch y 2")
    #
    #
    # print("generating")
    data, peaks = generate(n=500, rectified=False)
    # n = len(data)
    #
    #
    # plt.subplot(615)
    #
    # plt.plot(data[1])
    # plt.title("spectrum")
    # plt.tight_layout()
    #
    #
    # plt.subplot(616)
    # plt.plot(Y)
    # plt.title('derectified spectrum')
    #plt.show()