
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.covariance import log_likelihood
import time
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
    def __init__(self, pathToData = None, data=None) -> None:
        self.data = parseData.parse(pathToData) if pathToData else data
        self.peaks = []
        self.neighborhoodSize = None
        self.kde = None
        self.likelihoods = None
        self.probabilities = None
        self.entropies = None
        self.meanEntropy = None
        self.sdEntropy = None
        self.regionalEntropy = None
        self.outliers = None
        self.tau = None
        self.alpha = None
    
       

        #Outliers of regional/smoothed entropy:
        #self.entropyOutliers, self.entropyTau = self.findOutliers(data=self.regionalEntropy, alpha = alpha)
        
        

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
    def setData(self, data:pd.DataFrame) -> None:
        self.data = data
    def setKDE(self, kde:object) -> None:
        self.kde = kde
    def setLikelihoods(self, likelihoods:np.array) -> None:
        self.likelihoods = likelihoods
    def setProbabilities(self, probabilities:np.array) -> None:
        self.probabilities = probabilities
    def setEntropies(self, entropies:np.array) -> None:
        self.entropies = entropies
    def setMeanEntropy(self, meanEntropy:np.float128) -> None:
        self.meanEntropy = meanEntropy
    def setSDEntropy(self, sd:np.float128) -> None:
        self.sdEntropy = sd
    def setRegionalEntropy(self, regionalEntropy:np.array) -> None:
        self.regionalEntropy = regionalEntropy
    def setOutliers(self, outliers:np.array) -> None:
        self.outliers = outliers
    def setTau (self, tau:np.float128) -> None:
        self.tau = tau
    def setEntropyOutliers (self, entropyOutliers) -> None:
        self.entropyOutliers = entropyOutliers
    def setEntropyTau(self, tau) -> None:
        self.entropyTau = tau





















    
    ###################################################################################
    # METHODS:
    ###################################################################################
    def findPeaks(self, alpha=0.94, neighborhoodSize=4, entropyRegionSize = 1, method:str="palshikar")->None:
        """TO WRITE:
        A wrapper function that calls the appropriate algorithm. It then sets self.peaks to the output value of the method called
        """
        self.alpha = alpha
        t0 = time.perf_counter()
        methods = ["palshikar"]
        if method not in methods:
            raise ValueError(f"{method} is not an accepted method.\nAccepted methods: {methods}")
        elif method == "palshikar":
            self.neighborhoodSize = neighborhoodSize

            if not self.kde:
                #print("Fitting Kernel Density Estimate (This can take a while with large datasets)...")
                t1 = time.perf_counter()
                self.setKDE(self.estimateKernel())
                t2 = time.perf_counter()
                #print(f"Done. Took {t2-t1} seconds.")
            if self.likelihoods is None:
                #print('Calculating Entropy Values...')
                t1 = time.perf_counter()
                self.setLikelihoods(self.kde.score_samples(np.array(self.data[1]).reshape(-1, 1)))
                self.setProbabilities(np.float128(np.exp(self.likelihoods)))
                self.setEntropies(np.float128(-self.probabilities*self.likelihoods))
                self.setMeanEntropy(np.mean(self.entropies))
                self.setSDEntropy(np.std(self.entropies))
                self.setRegionalEntropy(self.findRegionsOfHighEntropy(width = entropyRegionSize))
                t2 = time.perf_counter()
                #print(f"Done. Took {t2-t1} seconds.")
            if self.outliers is None:
                #print("Finding outlier points...")
                t1 = time.perf_counter()
                outliers, tau = self.findOutliers(data=self.regionalEntropy, alpha = alpha)
                self.setOutliers(outliers)
                self.setTau(tau)
                t2 = time.perf_counter()
                #print(f"Done. Took {t2-t1} seconds")
            #print("Running peak finding algorithm...")
            t1 = time.perf_counter()
            self.setPeaks(self.findPeaksFromEntropyOutliers(self.outliers, threshold = neighborhoodSize))
            t2 = time.perf_counter()
            #print(f"Finished search. Total time: {t2-t0} seconds.")
            #print(f"Found {len(self.peaks)} peaks.")


    def estimateKernel(self) -> object:
        """
        ###This is lagging hard, probably due to gridsearch. need to optimize
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
        h_vals = np.arange(0.05, 1, 0.3)
        #Kernel types to evaluate:
        kernels = ['epanechnikov']

        #Optimize h (smoothing parameter) and kernel type:
        grid = GridSearchCV(KernelDensity(),
    {'bandwidth': h_vals, 'kernel': kernels},
    scoring=my_scores)
        grid.fit(np.array(self.data[1]).reshape(-1, 1))
        #save best model
        kde = grid.best_estimator_
        return kde

    def findOutliers(self, data:np.array, alpha:float ):
        """
        #Should refactor this to be a general outlier detector with a argument
        This function calculates the alpha-th quantile of the smoothed entropies. It uses this to filter out the 'normal' data, and returns the indices of the outliers

        alpha: what quantile to calculate

        Returns: np array of indices of outliers
        """
        #caluclate the alpha-quantile of the spectrum:
        tau = mquantiles(data, 1. - alpha)
        #this finds indexes where likelihood is less than tau, which then needs to be flattend because of the output format:
        outliers= np.argwhere(data < tau).flatten()
        while len(outliers) < 1:
            #If no outliers were found, adjust alpha downwards until some are detected:
            alpha = 0.9*alpha
            tau = mquantiles(data, 1. - alpha)
            outliers= np.argwhere(data < tau).flatten()

        return outliers, tau

   
    

    ######
    #Now, we have an object with a list of entropy differentials, and we just need to run the algorithm across that list. This should reduce the complexity from O(Length*WindowWidth) to O(Length+Length)
    #####
    

    def findRegionsOfHighEntropy(self, width:float) -> list:
        """
        width: how wide of a band of wavelengths should be considered. this many right and left neigbours will be considered
        """
        if width == 0:
            return self.entropies
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
        """
        What is the point of this? what should it show? 
        """
        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.plot(self.data[1])
        plt.title("Spectrum")
        if showPeaks:
            peakValues = [self.data[1][i] for i in self.peaks]
            ##print(peakValues)
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
        
        
        totalError = 0
        peaks_found = [0 for peak in true]
        for estimatedPeak in estimated:
            currentDistance = np.inf
            for index, truePeak in enumerate([*true, -np.inf]):
                if currentDistance >= np.abs(estimatedPeak-truePeak):
                    currentDistance = np.abs(estimatedPeak-truePeak)
                else: #this peak is a worse fit than the prior, save distance and break out of inner loop
                    if currentDistance <= 5:
                        peaks_found[index-1] = 1
                    totalError += currentDistance
                    break
        averageError = (totalError / (len(estimated) * len(self.data)))
        return (1-averageError, np.sum(peaks_found)/len(true))

    def bootstrap(self, estimated, real, n, sample_size = 100):
        """
        To do-----
        """
        scores = []
        for i in range(sample_size): 
            scores.append( self.assess(np.random.randint(0,n,len(estimated)), real))
        estimate_score = self.assess(estimated, real)[0]
        scores.append(estimate_score)
        scores.sort()
        
        return (scores.index(estimate_score) / sample_size)



    def findPeaksFromEntropyOutliers(self, data:np.array, threshold:int = 10) -> list:
        """
        ###CONSIDER REFACTORING using a while loop that runs descend and explore several times###
        This function implements a simple peak finding algorithm on outlier values of Shannon's entropy values. First it calls descend() on the outliers which is itself a simple algorithm to find the value closest to a local peak for each value in a set of indices. Then, in the nested list comprehension it runs explore() on the local peaks estimates. This function looks for a local peak in the neighborhood of a given point. The list comp then filters out duplicates. Finally, findGroups() groups together points in case there are multiple estimates for a single peak (i.e. closer than a threshold value), and averagePeaks averages those index values to arrive at a better estimate.

        data: array of indices to run the algorithm on. They serve as "seeds" to start the search for peaks. For best results, they should be close to peak regions.

        threshold: maximum distance between two points to be considered part of the same peak

        returns: list of estimated peak indices.
        """
        #find local optima closest to outlier points:
        outliers = self.descend(self.outliers)
        peaks = []
        #this nested list comprehension does several things: 1) flattens the output of repeated calls to explore into a 1d list
        #2) filters out duplicates, and 3) appends the results to the list of peaks
        [peaks.append(index) for result in [self.explore(point, window=self.neighborhoodSize) for point in outliers] for index in result if index not in peaks]
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
            return np.array([int(peak) for peak in averagedPeaks if peak == peak])
        groups = findGroups(peaks)
        averagedPeaks = averagePeaks(groups)
        
        return averagedPeaks


    
"""
To Do:

[X] Clean up the code, get rid of unnecessary functions
[X] Assess needs to raise a red flag if a true peak is missed
[X] look for duplicated calculations
    [X] distance matrix?
[] look into a simple way to get outliers

[] make the .show() method cleaner, and use code from below
[X] Time each segment of the algorithm with data of varying lengths
[X] test with larger inputs (time it to see how it varies with n)
[X] test with more noise
[X] Test the accuracy of the algorithm with varying:
    [X] NeighborhoodSize (used in the explore subroutine to merge close peaks)
    [X] alpha (used in the outlier search)
[] Learn about Virtual Environments in python
[] set up virtual environment for this project
[] implement version of peakfinder that calculates kde for segments of the curve
[] implement version of peakfinder that calculated 2d kde for curve
[] implement non-rectified data generation
[] test algorithm on non-rectified data
"""

if __name__ == '__main__':
    data, realpeaks = generate(n=2000, noise = 10)
    realpeaks.sort()

    spectrum1 = Spectrum(None, data=data)
    spectrum1.findPeaks(method="palshikar", alpha = 0.96, entropyRegionSize=2)
  

    #initiate plot
    plt.subplots(2,1)
    plt.subplot(211)
    #show points found using outlier method
    plt.plot(spectrum1.data[1])
    plt.axvline(realpeaks[0], c='red', alpha=.6, label="True Lines")
    [plt.axvline(peak, c='red', alpha=0.6) for peak in realpeaks]

    peaks = spectrum1.getPeaks()
    plt.scatter(x=peaks, y = spectrum1.data[1][peaks], c = "black", alpha = .8, label = "Inferred Lines")
    plt.ylabel('Intensity')
    plt.title('Simulated non-noisy Spectrum showing true and inferred peaks')
    plt.legend(loc = "upper right")

    #regional entropy:
    plt.subplot(212)
    plt.plot(spectrum1.regionalEntropy)
    plt.axhline(spectrum1.tau, c = "red", alpha=0.6, label = "tau")
    plt.title(f"Smoothed Entropies for Outlier Detection (alpha = {spectrum1.alpha})")
    plt.ylabel("Shannon\'s Entropy")
    plt.xlabel("\'Wavelength\'")
    plt.legend()
    #likelihoods:

    
    plt.tight_layout()
    plt.show()




    