from timeit import repeat
from matplotlib.cbook import index_of
import numpy as np, pandas as pd, math, matplotlib.pyplot as plt
from sklearn.covariance import log_likelihood
import parseData
from tables import Float32Atom, test
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV




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

    def calculateEntropyDifference(self, index:int) -> float:
        """
        Calculates the entropy of the sequence without the value at index.

        index: integer index of the wavelength to find the entropy of.
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


        




    
"""
To Test:
[] Instantiate a class
[] do all features work
    [] get Peaks
    [] estimate Kernel
    [] evaluate KDE
    [] calculate log2 of array
    []
[] does the algorithm find peaks?
[] can it handle large inputs
"""