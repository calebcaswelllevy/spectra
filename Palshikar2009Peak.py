
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.covariance import log_likelihood

import parseData

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import OPTICS
from generateData import generate
from scipy.stats.mstats import mquantiles


class Spectrum:
    """
    #the current sensitivity and width params are used only in the first implementation of hte algorithm. should refactor this so that the newer outlier approach takes these params. Also, should consider not running the peak finding algorithm until a method is called.  Also need to make sure all the attributes are being used and are not being duplicated. also, make all the getters and setters.
    ####################
    Spectrum object holds data for a spectrum. It has methods to implements a peak finding algorithm.
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

    ###################################################################################
    # GETTERS:
    ###################################################################################

    def getPeaks(self) -> list:
        return self.peaks
    def getData(self) -> pd.DataFrame:
        return self.data
    def getKDE(self) -> object:
        return self.kde
    def getLikelihoods(self) -> np.array:
        return self.likelihoods    
    def getProbabilities(self) -> np.array:
        return self.probabilities
    def getEntropies(self) -> np.array:
        return self.entropies
    def getMeanEntropy(self) -> np.float128:
        return self.meanEntropy
    def getSDEntropy(self) -> np.float128:
        return self.sdEntropy
    def getRegionalEntropy(self) -> np.array:
        return self.regionalEntropy
    def getOutliers(self) -> np.array:
        return self.outliers
    def getTau (self) -> np.float128:
        return self.tau
    def getEntropyOutliers (self) -> np.array:
        return self.entropyOutliers
    def getEntropyTau(self) -> np.float128:
        return self.entropyTau

    ###################################################################################
    # SETTERS:
    ###################################################################################

    def setPeaks(self, peaks:list) -> None:
        self.peaks = peaks
    
    ###################################################################################
    # METHODS:
    ###################################################################################

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

        Returns:  probability density estimated by the KDE.
        """
        return np.float128(np.exp(self.kde.score_samples(np.array(self.data[1]).reshape(-1, 1))))

    def calculateLog2OfArray(self) -> np.array:
        """
        #this is only needed to calculate binary entropy, can refactor to use e and get rid of this function.

        Calculates the log2 of the array of probability densities
        """
        return np.float128(np.log2(self.probabilities))
       
        

    def entropy2(self, start:int, stop:int) -> float:
        """
        #refactor this to use the ln values given by the kde

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
        #Might not need this one, see if anything is using it
        Calculates the entropy of the sequence without the value at index.

        index: integer index of the wavelength to find the entropy of.
        """
        return -self.entropies[index]

    def findOutliers(self, alpha:float = 0.90) -> np.array:
        """
        #should merge this with findSmoothedEntropyOutliers, see note below
        This function calculates the alpha-th quantile of the KDE log Likilihoods. It uses this to filter out the normal data, and returns the indices of the outliers.py

        alpha: what quantile to calculate

        Returns: np array of indices of outliers
        """
        #caluclate the alpha-quantile of the spectrum:
        tau = mquantiles(self.likelihoods, 1. - alpha)
        #this finds indexes where likelihood is less than tau, which then needs to be flattend because of the output format:
        outliers= np.argwhere(self.likelihoods < tau).flatten()
        return outliers, tau

    def findSmoothedEntropyOutliers(self, alpha = 0.95):
        """
        #Should refactor this to be a general outlier detector with a argument
        This function calculates the alpha-th quantile of the smoothed entropies. It uses this to filter out the 'normal' data, and returns the indices of the outliers

        alpha: what quantile to calculate

        Returns: np array of indices of outliers
        """
        #caluclate the alpha-quantile of the spectrum:
        tau = mquantiles(self.regionalEntropy, 1. - alpha)
        #this finds indexes where likelihood is less than tau, which then needs to be flattend because of the output format:
        outliers= np.argwhere(self.regionalEntropy < tau).flatten()
        return outliers, tau

    ######
    #Now, we have an object with a list of entropy differentials, and we just need to run the algorithm across that list. This should reduce the complexity from O(Length*WindowWidth) to O(Length+Length)
    #####
    def evaluatePeaks(self, sensitivity:float, width:float) -> list:
        """
        #CAN PROBABLY DISCARD THIS VERSION
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

        #initialize list of entropies to the average (Values from 0:width and n-width:n will remain at the average to avoid NaN):
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
    def descend(self, points:np.array) -> np.array:
        """
        A simple hill climbing algorithm to find all peaks in a set of points by traversing left to right through all points.

        points: an np array or list of the indices of the points in the spectrum to be evaluated

        returns: np array of estimated peak indices
        """
        spectrum = self.data[1]
        peaks = []
        descending, index = True, 0
        while True:
            if descending and index + 1 == len(points):#descending and at and of data, ->peak, break out of loop
                peaks.append(points[index])
                break
            elif index + 1 == len(points):
                break
            elif descending and spectrum[points[index]] < spectrum[points[index+1]]:#at a peak
                peaks.append(points[index])
                descending = False
            elif not descending and spectrum[points[index]] >= spectrum[points[index+1]]:
                descending = True
            index += 1
        return np.array(peaks)

    def explore(self, point:int, window:int = 5) -> int:
        """
        A simple algorithm to find the lowest point within a neighborhood window.

        point: index of starting point to explore from

        returns: index or indices of lowest point found during exploration
        """
        
        #handle edge cases:
        if point+window+1 > len(self.data[1]):
            neighborhood = self.data[1][point-window:len(self.data[1])]
            indices = [i for i in range(point-window, len(self.data[1]))]
        elif point < window:
            neighborhood = self.data[1][0:point+window+1]
            indices = [i for i in range(point+window+1)]
        else:  #neighborhood is not on the edge:   
            neighborhood = self.data[1][point-window:point+window+1]
            indices = [i for i in range(point-window, point+window+1)]
        minimum = min(neighborhood)
        #in case of a tie, returns list with all indices:
        
        return [index for index, value in zip(indices, neighborhood) if value == minimum]


    #could also use HDBSCAN to find clusters, and use cluster centroids as starting points
    def clusterOutliers(self, data: list) -> object:
        """
        #this can probably be removed, but should definitily be refactored to be more of a general tool if I keep it. The simpler distance based algorithm seems to be better for this purpose.

        clusters a one dimensional data set.

        returns: optics model object
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

    def assess(self, estimated:list, true:list) -> float:
        """
        A function to assess the fit of the estimated peaks to the true peaks when synthetic data is used. This funciton assumes both lists are sorted.

        estimated: a list of estimated peak indices
        true: a list of true peak locaitons

        returns: a float average error 
        """
        distanceMatrix = np.abs(estimated[:, None] - true[None, :])


        totalError = 0
        for estimatedPeak in estimated:
            currentDistance = np.inf
            for truePeak in true:
                if currentDistance > np.abs(estimatedPeak-truePeak):
                    currentDistance = np.abs(estimatedPeak-truePeak)
                else: #this peak is a worse fit than the prior, save distance and break out of inner loop
                    print(f"estimated: {estimatedPeak}. matched to: {truePeak}")
                    totalError += currentDistance
                    break
        averageError = totalError / len(estimated)
        return averageError

    def findPeaksFromEntropyOutliers(self, threshold:int = 10) -> list:
        """
        This function implements a simple peak finding algorithm on outlier values of Shannon's entropy values. First it calls descend() on the outliers which is itself a simple algorithm to find the value closest to a local peak for each value in a set of indices. Then, in the nested list comprehension it runs explore() on the local peaks estimates. This function looks for a local peak in the neighborhood of a given point. The list comp then filters out duplicates. Finally, findGroups() groups together points in case there are multiple estimates for a single peak (i.e. closer than a threshold value), and averagePeaks averages those index values to arrive at a better estimate.

        threshold: maximum distance between two points to be considered part of the same peak

        returns: list of estimated peak indices.
        """
        
        outliers = self.descend(self.entropyOutliers)
        peaks = []
        #this nested list comprehension does several things: 1) flattens the output of repeated calls to explore into a 1d list
        #2) filters out duplicates, and 3) appends the results to the list of peaks
        [peaks.append(index) for result in [self.explore(point) for point in outliers] for index in result if index not in peaks]
        peaks = np.array(peaks)
        #now need to find groups and average them:
        #Threshold might need to be optimized for each dataset:
        def findGroups(peaks:np.array):
            """
            This is an accessory function to group together peak estimates that are closer than a threshold value given as an argument in the parent function.

            peaks: array of peak indices.

            Returns: nested list of grouped indices
            """       
            groups = []
            rowsToSkip = []
            distanceMatrix = np.abs(peaks[:, None] - peaks[None, :])
            for index, row in enumerate(distanceMatrix):
                if index in rowsToSkip: 
                    groups.append([])
                    pass
                else:
                    
                    groups.append([index])
                    for index2, distance in enumerate(row):
                        if distance < threshold and index2 not in groups[index]:
                            groups[index].append(index2)
                            rowsToSkip.append(index2)
            return groups
        def averagePeaks(groups:list) -> list:
            """
            This is an accessory function that averages grouped index values (using integer division). 

            groups: a nested list of indices to be averaged (within subgroups)

            Returns: flat list of averages for each sublist
            """
            averagedPeaks = []
            for group in groups:
                subgroup = []
                for index in group:
                    subgroup.append(peaks[index])
                mean = np.floor(np.mean(subgroup))
                averagedPeaks.append(mean)
            return [int(peak) for peak in averagedPeaks if peak == peak]
        groups = findGroups(peaks)
        averagedPeaks = averagePeaks(groups)

        print(groups)
        print(peaks)
        print("Score:", self.assess(peaks, self.data[1]))
        print(averagedPeaks)
        print("Score2", self.assess(averagedPeaks, self.data[1]))
        
        return averagedPeaks


    
"""
To Do:

[] Clean up the code, get rid of unnecessary functions
[] look for duplicated calculations
    [] distance matrix?
[] look into a simple way to get outliers
[] test with larger inputs (time it to see how it varies with n)
[] test with more noise
[] make the .show() method cleaner, and use code from below
[] Time each segment of the algorithm with data of varying lengths
"""

if __name__ == '__main__':
    data, realpeaks = generate(n=2000, noise = 4)


    spectrum1 = Spectrum(None, data=data, sensitivity=2)
  

    #initiate plot
    plt.subplots(5,1)
    plt.subplot(511)
    #show points found using outlier method
    plt.plot(spectrum1.data[1])
    #plt.scatter(x = spectrum1.outliers, y = spectrum1.data[1][spectrum1.outliers], c='red', alpha = 0.3)
    model = spectrum1.clusterOutliers(spectrum1.entropyOutliers)
    clusters = model.labels_
    plt.scatter(x=spectrum1.entropyOutliers, y = spectrum1.data[1][spectrum1.entropyOutliers], c =clusters, alpha = 0.5, cmap='Set1')
    plt.title("Spectrum showing outliers")

    #points found using descend algorithm on outliers:
    plt.subplot(512)
    peaks = spectrum1.descend(spectrum1.entropyOutliers)
    plt.plot(spectrum1.data[1])
    plt.scatter(x=peaks, y = spectrum1.data[1][peaks], c = "red", alpha = .8)
    plt.title('Peaks found using climb algorithm on outliers')

    #points found using explore algorithm on descend points:
    plt.subplot(513)

    peaks2 = []
    peaks2 = [index for result in [spectrum1.explore(point) for point in peaks] for index in result if index not in peaks2]
    
    #peaks = [peak for peak in peaks if peak not in peaks]

    #peaks = [index for result in [spectrum1.explore(point) for point in peaks] for index in result]
    print("Found peaks: \n", peaks)
    #[print(spectrum1.data[1][peak]) for peak in peaks]
    plt.plot(spectrum1.data[1])
    plt.scatter(x=peaks, y = spectrum1.data[1][peaks], c = "red", alpha = .8)
    plt.title('Peaks found using explore algorithm on descend points')

    #regional entropy:
    plt.subplot(514)
    plt.plot(spectrum1.regionalEntropy)
    plt.axhline(spectrum1.entropyTau, c = "red", alpha=0.6)
    plt.title("Smoothed Entropies")

    #likelihoods:
    plt.subplot(515)
    plt.plot(spectrum1.likelihoods)
    plt.axhline(spectrum1.tau, c = "red", alpha=0.6)
    plt.title("KDE Likelihoods and tau")

    
    plt.tight_layout()
    #plt.show()

    
        
    spectrum1.findPeaksFromEntropyOutliers()
    print(spectrum1)