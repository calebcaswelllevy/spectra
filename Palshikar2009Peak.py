from operator import ne
from timeit import repeat
from matplotlib.cbook import index_of
import numpy as np, pandas as pd, math, matplotlib.pyplot as plt
from sklearn.covariance import log_likelihood
from sqlalchemy import true
import parseData

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import OPTICS
from generateData import generate
from scipy.stats.mstats import mquantiles





class Spectrum:
    """
    Spectrum object holds data for a spectrum. It implements the peak finding algorithm
    """
    def __init__(self, pathToData = None, data=None, sensitivity = 0.6, width=10) -> None:
        self.data = parseData.parse(pathToData) if pathToData else data
        self.kde = self.estimateKernel()
        self.peaks = []
        self.likelihoods = self.kde.score_samples(np.array(self.data[1]).reshape(-1, 1))
        
        #I am duplicating the score calculation here, can rework this:
        self.probabilities = np.float128(self.evaluateKDE())

        #Should remove this step and use the ln values given by the kde:
        #self.log2Likelihoods = np.float128(self.calculateLog2OfArray())

        self.entropies = np.float128(-self.probabilities*self.likelihoods)
        self.meanEntropy = np.mean(self.entropies)
        self.sdEntropy = np.std(self.entropies)
        self.peaks = self.evaluatePeaks(sensitivity, width)
        self.regionalEntropy = self.findRegionsOfHighEntropy(width = 2)
        #To implement:
        #outlier score and tau (cutoff value)
        self.outliers, self.tau = self.findOutliers(alpha = 0.92)

        #Outliers of regional/smoothed entropy:
        self.entropyOutliers, self.entropyTau = self.findSmoothedEntropyOutliers(alpha = 0.94)
        
        #naive peakfinder
        #combined methods...
        ## Hillclimbing method from entropy peaks

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
        grid.fit(np.array(self.data[1]).reshape(-1, 1))
        #save best model
        kde = grid.best_estimator_
        return kde

    def evaluateKDE(self) -> np.array:
        """
        Evaluates the scaled probability density at each point in the spectrum
        """
        return np.float128(np.exp(self.kde.score_samples(np.array(self.data[1]).reshape(-1, 1))))

    def calculateLog2OfArray(self) -> np.array:
        """
        Calculates the log2 of the array of probability densities
        """
        return np.float128(np.log2(self.probabilities))
       
        

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
            entropy += np.float128(probability) * np.float128(np.log2(probability))

        return -entropy

    def calculateEntropyDifference(self, index:int) -> float:
        """
        Calculates the entropy of the sequence without the value at index.

        index: integer index of the wavelength to find the entropy of.
        """
        return -self.entropies[index]

    def findOutliers(self, alpha:float = 0.90) -> np.array:
        """
        This function calculates the alpha-th quantile of the KDE log Likilihoods. It uses this to filter out the normal data, and returns the indices of the outliers.py

        alpha: what quantile to calculate

        Returns: np array of indices of outliers
        """
        #caluclate the alpha-quantile of the spectrum:
        tau = mquantiles(self.likelihoods, 1. - alpha)
        #this finds indexes where likelihood is less than tau, which then needs to be flattend because of the output format:
        outliers= np.argwhere(self.likelihoods < tau).flatten()
        print("Outliers found at locations: ", outliers)
        return outliers, tau

    def findSmoothedEntropyOutliers(self, alpha = 0.95):
        """
        This function calculates the alpha-th quantile of the smoothed entropies. It uses this to filter out the normal data, and returns the indices of the outliers.py

        alpha: what quantile to calculate

        Returns: np array of indices of outliers
        """
        #caluclate the alpha-quantile of the spectrum:
        tau = mquantiles(self.regionalEntropy, 1. - alpha)
        #this finds indexes where likelihood is less than tau, which then needs to be flattend because of the output format:
        outliers= np.argwhere(self.regionalEntropy < tau).flatten()
        print("Outliers found at locations: ", outliers)
        return outliers, tau

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
        peaks = [index for index, entropy in enumerate(self.entropies) if -(entropy - self.meanEntropy) > (sensitivity * self.sdEntropy)]

        #Retain only one peak out of any set of peaks within k distance of each other:
        #Need to think about this and make it better:
        #print(f"peaks found: {peaks}")
        filteredPeaks = []
        startingPoint = 0
        while startingPoint < len(peaks):
            
            endingPoint = 1

            currentGroup = {peaks[startingPoint]: self.data[1][peaks[startingPoint]]}

            #find the group within k distance:
            while ((startingPoint+endingPoint) < len(peaks)) and (np.abs(peaks[startingPoint] - peaks[startingPoint+endingPoint]) <= width):
                currentGroup[peaks[startingPoint+endingPoint]] = self.data[1][peaks[startingPoint+endingPoint]]

                endingPoint += 1
            #find the bigest member of the group and add it to the list:
            filteredPeaks.append(max(currentGroup))

            
            #increment up the window:
            startingPoint += endingPoint
    

        print(f"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\nFound the following {len(filteredPeaks)} peaks: \n")
        [print(i, " - ", self.data[1][i]) for i in filteredPeaks]
        return filteredPeaks

    def findRegionsOfHighEntropy(self, width:float) -> list:
        """
        width: how wide of a band of wavelengths should be considered. this many right and left neigbours will be considered
        sensitivity: How far from mean should a wavelength be in order to be counted.
        """
        #the starting locations of the sliding window:
        center = width + 1
        left = 0
        right = 2 * width + 1

        #initialize
        regions = [self.meanEntropy for i in range(len(self.data))]
        while right <= len(self.entropies):
            region = self.entropies[left:right]
            
            regions[center] = np.mean(region)
    
            center += 1
            left += 1
            right += 1
        return regions

    #An algorithm that starts with outlier data and finds peaks using hillclimbing:
    #could also use HDBSCAN to find clusters, and use cluster centroids as starting points
    def climb(self, points, visibility=2):
        """
        """
        spectrum = self.data[1]
        peaks = []
        ascending, index = True, 0
        while index < len(points):
            if ascending and index + 1 == len(points):#ascending and at and of data, ->peak, break out of loop
                peaks.append(points[index])
                break
            elif ascending and spectrum[points[index]] > spectrum[points[index+1]]:#at a peak
                peaks.append(points[index])
                ascending = False
            elif not ascending and spectrum[points[index]] <= spectrum[points[index+1]]:
                ascending = True
            index += 1
        return peaks

        

    #could also use HDBSCAN to find clusters, and use cluster centroids as starting points
    def clusterOutliers(self, data):
        """
        clusters a one dimensional data set
        """
        model = OPTICS(min_samples=4).fit(data.reshape(-1, 1))
        return model

    def show(self, showPeaks = True, showEntropy = True):
        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.plot(self.data[1])
        plt.title("Spectrum")
        if showPeaks:
            peakValues = [self.data[1][i] for i in self.peaks]
            #print(peakValues)
            plt.scatter(self.peaks, peakValues, c='red')
        if showEntropy:

            plt.subplot(2,1,2)
            plt.plot(self.logLikelihoods)
            [plt.axvline(self.peaks[i], c='red', alpha=0.5) for i in range(len(self.peaks))]
        plt.show()


    
"""
To Test:
[x] Instantiate a class
[] do all features work
    [x] get Peaks
    [x] estimate Kernel
    [x] evaluate KDE
    [x] calculate log2 of array
    []
[] does the algorithm find peaks?
[] can it handle large inputs
"""
if __name__ == '__main__':
    data, peaks = generate(n=2000, noise = 2)

    #print(data[1])
    spectrum1 = Spectrum(None, data=data, sensitivity=2)
    #spectrum1.show(showPeaks=True)
    #[print(spectrum1.entropies[peak]) for peak in peaks]
    truePeaks = 0
    falsePeaks = 0
    for peak in spectrum1.peaks:
        if peak in peaks:
            truePeaks += 1
        else: falsePeaks += 1
    print(f'Correctly identified {truePeaks/len(peaks)}% of peaks')
    print(f'Falsely identified {falsePeaks/len(spectrum1.peaks)}')
    plt.subplots(3,1)
    plt.subplot(311)
    plt.plot(spectrum1.data[1])
    #plt.scatter(x = spectrum1.outliers, y = spectrum1.data[1][spectrum1.outliers], c='red', alpha = 0.3)
    model = spectrum1.clusterOutliers(spectrum1.entropyOutliers)
    clusters = model.labels_
    plt.scatter(x=spectrum1.entropyOutliers, y = spectrum1.data[1][spectrum1.entropyOutliers], c =clusters, alpha = 0.5, cmap='Set1')
    plt.title("Spectrum showing outliers")
    plt.subplot(312)
    plt.plot(spectrum1.regionalEntropy)
    plt.axhline(spectrum1.entropyTau, c = "red", alpha=0.6)
    plt.title("Smoothed Entropies")
    plt.subplot(313)
    plt.plot(spectrum1.likelihoods)
    plt.axhline(spectrum1.tau, c = "red", alpha=0.6)
    plt.title("KDE Likelihoods and tau")
    plt.tight_layout()
    plt.show()

    

   