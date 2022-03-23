import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def findPeak(spectrum:np.array, peakFunction:object = "entropyPeakFunction", windowSize:int = 10, scalerParam:float = .5):

    """
    Palshikar 2009 Simple Algorithms for Peak Detection in Time-Series

    implements simple algorithm for peak detection given a peak function.

    windowSize: determines the resolution of the peak size, in other words what is the minimum distance between two peaks for them to be considered separate

    scalerParam: how many standard deviations away from mean peakFunction value is weird enough to be a peak. Lower number should result in more peaks found, higher number less false positives. 
    """

    peaks = []
    a = pd.Series([0.0 for i in range(len(spectrum))])
    
    
    #compute peak function for each value
    print("at step 1")
    for index, wavelength in enumerate(spectrum):
        a[index] = peakFunction(2, 5, index, wavelength, spectrum)
    print("Output of peakFunction = ", a)

    #compute mean and standard dev for peak functions
    mean = np.mean(a[a>=0])
    sd = np.std(a[a>=0])
    print("Mean = ", mean)
    print("SD = ", sd)

    #remove locally small peaks (in context of overall spectrum)
    for index, metric in enumerate(a):
        if (metric > 0) and ((metric - mean) > (scalerParam * sd)):
            print("appending: ", index)
            peaks += [index]
            print(peaks)
    print("\n\nat step 2:")
    print("Peaks Found: ",peaks)
    
    """
    #NEED TO REWRITE THIS:
    #Find largest peak in windows:
    for index in range(len(peaks)-1):
        if peaks[index]-peaks[index+1] <= windowSize:
            if spectrum[index] > spectrum[index+1]:
                a[index+1] = -np.inf
            else:
                a[index] = -np.inf
    """
    #return peak indices with smaller close peaks filterd out:
    plt.plot(spectrum)
    plt.plot(a)
    plt.show()
    return pd.Series(peaks)[[value > -np.inf for value in peaks]]

 def entropyPeakFunction(self, sliceSize:int, width:int, index:int, spectrum:np.array) -> float:

        """
        #This function is made obsolete by the new class-based implementation

        calculate the difference in the entropy of two sequences, N(k, i, t)
        and N'(k, i, t), to give an idea of how influential a given x is.
        
        sliceSize:  a given integer, number of right and left neighbors to consider. E.g. a sliceSize of 5 will mean a total length of 11 with the value and 10 without.
        width: given window size for use in the probability calculation (p())
        index: index of value to calculate entropy with and without
        x: Value to calculate entropy of
        spectrum: set of data to find entropy

        Returns: the entropy score of a given locus
        """

        """
        #THIS BLOCK SHOULD BE DELETABLE NOW:
        # k values above and below x without x: 
        lower = [*spectrum[i-k:i]]
        upper = [*spectrum[i+1:i+k+1]]
        
        Nprime = lower + upper
        #k values above and below x including x:
        N = lower + [x] + upper

        spectrumPrime = [*spectrum[:i],  *spectrum[i+1:]]
        """
        

        start = index - sliceSize
        print(f"Entropy slice Starting at index: {start} which is value: {spectrum[start]}")
        entropyWithValue = entropy(start = index-sliceSize, length = 2 * sliceSize + 1, width = width, spectrum = spectrum)
        print(f"Entropy of sequence with value is: {entropyWithValue}")
        entropyWithoutValue = entropy(start = index-sliceSize, length = 2 * sliceSize, width = width, spectrum = [*spectrum[0:index]] + [*spectrum[index+1:]])

        print(f"Entropy of sequence with value is: {entropyWithValue}")
        print(f"Entropy without value is: {entropyWithoutValue}")

        return entropyWithValue - entropyWithoutValue
        #return (entropy(N, w, spectrum) - entropy(Nprime, w, spectrumPrime))

def entropy( start:int, length:int,  width:int, spectrum:np.array) -> float:
    """
    Calculate and return the entropy of a sequence of values.
    
    A: A full array of values, a subset of which will be measured for entropy.
    width: supplied window width parameter for the p function. 
        Int greater than 0.
    start: index to start calculation of entropy at
    length: size of sequence to calculate entropy
    
    """

    h = 0

    print("Calculating entropy for this sequence: ", spectrum[start:start+length])
    for index in range(length):
        ###################################################Index + width problem
        firstTerm = p(spectrum[start + index], start + index, spectrum, width)
        if firstTerm != firstTerm:
            #print("NaN enountered in Entropy Calculation")
            pass
        else:
            h += -firstTerm * np.log2(firstTerm)
        

    return h

def p( value:float, index:int, spectrum:np.array, width:int = 5, h=0.15) -> float:

    """
    Calculate and return probability density at a, using gaussian kernel
    index is index of balue
    value is value at index.
    spectrum is vector of values.
    width is given window size to calculate kernel where width is integer > 0
    h is the smoothing parameter
    """

    M = len(spectrum)
    n = 2*width +1

    #this subtracts the value from the value width units above. If values are the same, it adds a "psuedocount of 1%" to avoid division by 0.
    if ((index + width) >= M) or ((index - width) <= 0):
        print(f"Avoided calculating p for index {index} with width {width}")
        return np.NaN

    multiplicationFactor = 1/(n * h)

    summation = 0
    
    #starting width units before index up to width units after index, calculate the gaussian kernel and add to summation

    print("made it")
    
    for i in range(index - width, index + width + 1, 1):
        
        #add up kernel values
        summation += gaussianKernel(x = np.abs(value - spectrum[i]), sigma=h**0.5)
    return multiplicationFactor * summation
    try: pass
    except:
        print("_________________________OUT OF BOUNDS ERROR_________________________")
    # [print(index, value) for index, value in enumerate(A)]
        print("index = ", index)
        print("width = ", width)
        print("Array size = ", M)
        print("if statement evaluates to: ", ((index - width) <= 0) or ((index + width - 1) >= M))
        print("index + width = ", index + width)
        return 0
    
 def gaussianKernel( x:float, sigma:float) -> float:
    """
    Calculates the gaussian kernel at a given locus

    x is a value for which to calculate the kernel
    """

    c = 1 / (((2 * math.pi)**0.5) * sigma)
    d =math.e**(-(x**2)/(2*(sigma**2)))
    print(f"First term of kernel is: {c}")
    print(f"second term of kernel is: {d}")
    print(f'x = {x}; sigma = {sigma}')
    print(f'This is what we are exponentiating e by: {(-(x**2)/(2*(sigma**2)))}')

    print(f"kernel for this locus is: {c * d}")
    return c * d

def epanechnikovKernel(x:float) -> float:
    """
    Calculates Epanechnikov Kernel for |x| <= 1 
    """
    #if value is less than 1, use Epanechnikov kernel:
    
    if (np.absolute(x) > 1):
        raise ValueError('Epanechnikov Kernel is only valid on values between -1 and 1')
    
    else:
        return (0.75 * (1 - (x**2)))


        