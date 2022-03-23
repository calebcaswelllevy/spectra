from timeit import repeat
from matplotlib.cbook import index_of
import numpy as np, pandas as pd, math, matplotlib.pyplot as plt
from sklearn.covariance import log_likelihood
import parseData
from tables import Float32Atom, test
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

class spectrum:
    def __init__(self, pathToData) -> None:
        self.data = parseData.parse(self.pathToData)
        self.kde = self.estimateKernel(self)
        self.peaks = []
        self.probabilities = self.evaluateKDE()
        #Should remove this step and use the ln values given by the kde:
        self.logLikelihoods = self.calculateLog2OfArray()
        self.entropies = -self.probabilities*self.logLikelihoods
        self.meanEntropy = np.mean(self.entropies)
        self.sdEntropy = np.std(self.entropies)
        self.peaks = self.evaluatePeaks()


    def getPeaks(self) -> list:
        return self.peaks
    def setPeaks(self, peaks:list) -> None:
        self.peaks = peaks

    def estimateKernel(self) -> object:
        """
        Method to estimate kernel density for spectrum. This is used to get probability density and entropy of segments of the spectrum.

        Returns: kernel density estimate model
        """
        def my_scores(estimator, X):
            """
            Custom scoring function to remove inf
            """
            scores = estimator.score_samples(X)

            #remove -inf:
            scores = scores[scores != float('-inf')]
            return np.mean(scores)

        #parameter vals to evaluate
        h_vals = np.arange(0.05, 1, 0.1)
        #Kernel types to evaluate:
        kernels = ['cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat']

        #Optimize h (smoothing parameter) and kernel type:
        grid = GridSearchCV(KernelDensity(),
    {'bandwidth': h_vals, 'kernel': kernels},
    scoring=my_scores)

        #save best model
        kde = grid.best_estimator_
        return kde

    def evaluateKDE(self) -> np.array:
        """
        Evaluates the scaled probability density at each point in the spectrum
        """
        return np.exp(self.kde.score_samples(self.data))

    def calculateLog2OfArray(self) -> np.array:
        """
        Calculates the log2 of the array of probability densities
        """
        return np.log2(self.probabilities)
       

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
        

    def entropy2(self, start:int, stop:int) -> float:
        """
        Using the probability densities from the Kernel Density Estimate, this function calculates the Shannon entropy of the slice within the range given by the indices: (start, stop].

        H(sequence) = sum(prob(element_i)*ln(prob(element_i)))

        Returns: the entropy of the sequence
        """

        probabilities = self.probabilities[start, stop]
        logLikelihoods = self.logLikelihoods[start, stop]
        entropy = 0
        for probability in probabilities:
            entropy += probability * np.log2(probability)

        return -entropy

    def calculateEntropyDifference(self, index) -> float:
        """
        Calculates the entropy of the sequence without the value at index.
        """
        return -self.entropies[index]


    

    ######
    #Now, we have an object with a list of entropy differentials, and we just need to run the algorithm across that list. This should reduce the complexity from O(Length*WindowWidth) to O(Length+Length)
    #####
    def evaluatePeaks(self, sensitivity:float, width:float) -> list:
        """
        This function loops through the entropy values, and saves the indices of values that are more than a certain multiple of Standard Deviations away from the mean. This is controlled by the sensitivity parameter. Higher sensitivity means a peak has to be farther from the mean to be found.

        Sensitivity: how many sds away from mean does a peak need to be. higher sensitivity means peaks must be bigger to be counted.

        Width: How far apart peaks are to be considered distict. A higher width parameter means less peaks.

        Returns: a list of  
        """

        #filter peaks out that are less than a certain number of SDs from the mean:
        peaks = [index for index, entropy in enumerate(self.entropies) if (entropy - self.meanEntropy) > (sensitivity * self.sdEntropy)]

        #Retain only one peak out of any set of peaks within k distance of each other:
        #Need to think about this and make it better:
        filteredPeaks = []
        startingPoint = 0
        while startingPoint < len(peaks):
            endingPoint = 1
            currentGroup = {peaks[startingPoint]: self.data[peaks[startingPoint]]}
            #find the group within k distance:
            while np.abs(peaks[startingPoint] - peaks[startingPoint+endingPoint]) <= width:
                currentGroup[peaks[startingPoint+endingPoint]] = self.data[peaks[startingPoint+endingPoint]]

                endingPoint += 1
            #find the bigest member of the group and add it to the list:
            filteredPeaks.append(max(currentGroup))
            
            #increment up the window:
            startingPoint += endingPoint
        
        return filteredPeaks


        




    def entropy(self, start:int, length:int,  width:int, spectrum:np.array) -> float:
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

    def p(self, value:float, index:int, spectrum:np.array, width:int = 5, h=0.15) -> float:

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
        
    def gaussianKernel(self, x:float, sigma:float) -> float:
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


            
"""
To Test:
[X] epanechnikovKernel
[X] gaussianKernel
[O] p
    [X] Handle divide by 0 problems
    [] handle window width (Might be a higher level problem)
[O] entropy
    [] Ensure calculation is correct
[] entropyPeakFunction
[] findPeak

"""

#gaussianKernel:

#1 uses Epanechnikov kernel works when below 1
#print(epanechnikovKernel(0) == .75)
#print(epanechnikovKernel(.999))

#2 epanechnikovKernel fails when 1 or higher:
#print(epanechnikovKernel(1))
#print(epanechnikovKernel(2))

#3 gaussian kernel returns probabilites
def testGaussian():
    print("\nTesting Gaussian Kernel\n")
    gammas =  [.25, .5, 1, 2, 3]
    x = [i/10 for i in range(100)]
    y = [[gaussianKernel(n-5, gamma=-g) for n in x] for g in gammas]
    print(y)
    print("y1 = ",y[1])
    for i in range(len(y)):
        plt.subplot(len(y), 1, i+1)
        plt.plot(x, y[i])
        plt.title(f'Kernel with gamma = {gammas[i]}')
    plt.show()

#4 test p():
def testP():
    print("\nTesting p():\n")


    
    testData1 = np.array([i % 10 for i in range(100)])
    testData2 = [i for i in range(50)]  
    testData2 += [i for i in range(50, 0, -1)]
    
    #does it work?
    #print(p(testData1[50], 50, testData1, 5))
    #print(p(testData1[49], 49, testData1, 5))

    #probability should be less at a peak than on a slop or flat:
    #print("Peak is less probable: ", p(testData2[50], 50, testData2, 5) < p(testData2[47], 47, testData2, 5))
    

    #does it throw an errow when index is closer to edge than width?
    """try:
        print(p(testData1[99], 99, testData1, 5))
    except:
        print("Throws error when window is too wide")"""


    prob = [0 for i in range(len(testData2)-20)]
    print(prob)
    for index in range(len(testData2)-20):
        prob[index] = p(value = testData2[index+10], index = index, spectrum = testData2, width = 5)
        #prob[index] = p(a=testData2[index+10], index = index, A = testData2, width = 5)
        print("INdex = ", index)
        print("data = ", testData2[index+10])
        print(f"p of {testData2[index+10]} = {prob[index]}")
        print()
        if prob[index] > 1: print("!"*170, f"\n{testData2[index+4]}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print()
    plt.subplot(2,1,1)
    plt.plot(prob)
    plt.subplot(2,1,2)
    plt.plot(testData2)
    print(prob)
    print(prob[40:60])
    print(prob[43])
    print(testData2[50:60])
    print(p(testData2[55],55,testData2, width = 5))
    plt.show()

    #
def testEntropy():

    small = np.random.rand(100)
    medium = np.random.rand(100)*10
    big = np.random.rand(100)*100
    
    es = entropy(small[10:21], 10, small)
    em = entropy(medium[10:21], 10, medium)
    eb = entropy(big[10:21], 10, big)

    print("small e: ", es)
    print("med e: ", em)
    print("big e: ", eb)

    #Does

    min =  [0,0,0,0,0,0,0,0,0,0]
    max = np.random.rand(10)*1000
    mid = [0,0,0,0,100,0,0,0,0,0]
    es = entropy(min, 10, min)
    em = entropy(max, 10, max)
    eb = entropy(mid, 10, mid)

def testEntropy():

    small = np.random.rand(100)
    medium = np.random.rand(100)*10
    big = np.random.rand(100)*100
    structured = list(small[0:30]) + list(medium[0:40]) + list(big[0:30])
    


    es = entropy(start = 0, length = len(small), width = 5, spectrum = small)
    em = entropy(start = 0, length = len(medium), width = 5, spectrum = medium)
    eb = entropy(start = 0, length = len(big), width = 5, spectrum = big)

    print("small e: ", es)
    print("med e: ", em)
    print("big e: ", eb)

    entropies = [0 for i in range(8)]
    length = 10
    for i in range(8):
        start = i*10
        print("starting at:", start)
        print("slice is: ", structured[start:start+length])
        entropies[i] = entropy(start, length, 5, structured)
        print(f"Entropy for this slice is: {entropies[i]}")
    print("Entropies:")
    print(entropies)






def testEntropyPeakFunciton():
    np.random.seed(1)
    testData1 = list(np.random.rand(9)) + [5, 10, 5] + list(np.random.rand(9))
    test1 = entropyPeakFunction(sliceSize = 5, width = 5, index = 10, spectrum = testData1)
    
    print(f"tested sequence should be: {testData1[5:16]}")
    print(f"the entropy differenece of the tested sequence was: {test1}")


def testPeakfinder():
    peakData = [0,2,1,0,1,2,3,4,100,4,3,2,1,0,0,2,2,0,0,0,1,2,3,4,100,4,3,2,1,0,-1,0,1]

    def simplerFunction(k=0, w=0, i=0, x=0, spectrum=peakData):
        print("i = ",i)
        print("x = ", x)
        print("w = ", w)
        print("spectrum = ", spectrum)
        if spectrum[w] == max(spectrum):
            return 100
        return np.random.rand()
    """
    Problems:
    need informative variable names
    w is index
    i is value
    x is the full vector
    spectrum is the full vector
    """
    #peak = findPeak(spectrum = peakData, peakFunction = simplerFunction, windowSize = 3, scalerParam = 2)
   

    peak = findPeak(spectrum = peakData, peakFunction = entropyPeakFunction, windowSize = 3, scalerParam = 2)
    print(peak)
    [print(index, ": ", value) for index, value in enumerate(peakData)]
            
    
#testGaussian()
testP()
#testEntropy2()
#testEntropyPeakFunciton()

#testPeakfinder()


