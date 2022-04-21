from cProfile import label
from generateData import generate
from Palshikar2009Peak import Spectrum
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from time import perf_counter_ns, perf_counter

#To test:
#[X] peak number
#[X] noise vs performance
#[X] Amount of data vs time
#[ ] Amount of data vs performance?
#[X] Alpha vs performance
#[X] neighborhood size
#[ ] relationship between noise and alpha
#[ ] relationship between noise and neighborhood size
#[ ] relationship between alpha and neighborhood size


neighborhood_sizes = np.arange(2,20, 1)

datasets = generate()

###########################################################
# Test 1: time vs n
###########################################################
def test_time():
    dataSizes = [int(np.floor(n)) for n in np.linspace(start=2000, stop=80000, num = 15)]
    times = []
    performances = []
    recalls = []
    for n in dataSizes:
        print('*'*150)
        print("starting test on size", n)
        data, peaks = generate(n)
        peaks.sort()
        t1 = perf_counter()
        s = Spectrum(data = data)

        s.findPeaks(alpha=0.95)
        t2 = perf_counter()
        times.append(t2-t1)
        performance, recall = s.assess(s.peaks, peaks)
        performances.append(performance)
        recalls.append(recall)
        
        print("Completed in ", t2-t1)

    plt.subplots(2,1)
    plt.subplot(211)

    plt.scatter(x=dataSizes, y = times)
    plt.title("Time (ns) vs. data size")
    plt.subplot(212)
    
    plt.scatter(x = dataSizes, y = performances, c="blue",alpha=0.6, label="performance")
    plt.scatter(x = dataSizes, y=recalls, c = "red", alpha=0.6, label="recall")
    plt.legend(loc="lower left")
    plt.title("Performance vs n")
    plt.ylim((0,1))
    plt.xlabel("Data Size")
    plt.tight_layout()
    plt.show()

"""
#Results:
1 - gridsearchCV is the culprit of the slow runtimes. 
2 - fitting using only the Epanichikov kernel rather than all kernels dramatically speeds things up
3 - increasing the density of the h search doesnt have a big effect.
4 - Now I need to test the performance of the algorithm with the epanichikov kernel
    - seems to work just fine with just the epanichikov kernel
5 - still runs slow with large inputs- the model fitting stage is something like O(n2). Linear binning might be worth looking into. Another option is to fit a subset of the data for the really big spectra
6 - Also, using that package with the fft implementation might be worthwile, if I can figure out how to optimize h parameter.

"""
###########################################################
# Test 2: peak number:
###########################################################
def TestPeakNumber():
    ranges = range(10, 1000, 50)
    datasets =  [generate(n=2000, nPeaks = i) for i in ranges]
    peakNo = [i for i in ranges]
    performances = []
    for data, peaks in datasets:
        peaks.sort()
        spectrum = Spectrum(data = data)
        spectrum.findPeaks()
        performances.append( spectrum.assess(spectrum.peaks, peaks))
    plt.scatter(x = peakNo, y = performances)
    plt.title("Algorithm performance vs peak density")
    plt.show()

    """
    RESULTS:
    1. Algorithm performs similarly with varying numbers of peaks. It performs better with many peaks but this may be simply due to decreased average distance to peaks. A better assessment metric might be useful (Percent found or similar)
    """

###########################################################
# Test 3: noise vs. performance
###########################################################
def testNoise():
    datasets = [generate(n=2000, nPeaks = 10, noise= i) for i in range(2, 100, 10)]
    performances = []
    noiselevel = [i for i in range(2, 100, 10)]
    for data, peaks in datasets:
        peaks.sort()
        spectrum = Spectrum(data = data)
        spectrum.findPeaks()
        performances.append(spectrum.assess(spectrum.peaks, peaks))
    plt.scatter(x = noiselevel, y = performances)
    plt.title("Performance vs. Noise Level")
    plt.show()

###########################################################
# Test 4: noise vs. performance
###########################################################
def testNoise(alpha=0.95, neighborhoodSize=5, entropyRegionSize=2):
    ranges = range(2, 100, 1)
    datasets = [generate(n=2000, nPeaks = 10, noise= i) for i in ranges]
    performances = []
    noiselevel = [i for i in ranges]
    for data, peaks in datasets:
        peaks.sort()
        spectrum = Spectrum(data = data)
        spectrum.findPeaks(alpha=alpha, neighborhoodSize=neighborhoodSize,entropyRegionSize=entropyRegionSize)
        performances.append(spectrum.assess(spectrum.peaks, peaks))
    plt.scatter(x = noiselevel, y = performances)
    plt.title("Performance vs. Noise Level")
    

###########################################################
# Test 5: alpha level vs. performance
###########################################################
def testAlpha():
    data, peaks = generate(noise = 10)
    peaks.sort()
    performances = []
    alphas = np.arange(0.9, .99, 0.1)
    for alpha in alphas:
        spectrum = Spectrum(data = data)
        spectrum.findPeaks(alpha=alpha)
        performances.append(spectrum.assess(spectrum.peaks, peaks))
    plt.scatter(x = alphas, y = performances)
    plt.title("Performance vs. alpha value")
    plt.show()

###########################################################
# Test 5: neighborhood size vs performance
###########################################################
def testNeighborhoodSize():
    data, peaks = generate(noise = 10)
    peaks.sort
    neighborhood_sizes = np.arange(2,20, 1)
    performances = []
    alphas = np.arange(0.9, .99, 0.1)
    for size in neighborhood_sizes:
        spectrum = Spectrum(data = data)
        spectrum.findPeaks(neighborhoodSize=size)
        performances.append(spectrum.assess(spectrum.peaks, peaks))
    plt.scatter(x = alphas, y = performances)
    plt.title("Performance vs. Neighborhood size")
    plt.show()
###########################################################
# Test 5: entropy region size vs performance
###########################################################

def test_entropy_region_size(replications = 10):
    datasets = []
    peaksets = []
    for i in range(replications):
        data, peaks = generate(noise = 10, nPeaks=30)
        datasets.append(data)
        peaksets.append(peaks)
        peaks.sort()
    entropy_region_sizes = [i for i in range(50)]
    performances, recalls = [], []
    for data, peaks in zip(datasets, peaksets):    
        for size in entropy_region_sizes:
            spectrum = Spectrum(data = data)
            spectrum.findPeaks(entropyRegionSize=size)
            print(spectrum.peaks)
            print(peaks)
            perf, rec = spectrum.assess(spectrum.peaks, peaks)
            performances.append(perf)
            recalls.append(rec)
          
    plt.scatter(x = entropy_region_sizes*replications, y = performances, c="blue", alpha=.4, s = 30,  label="performance")
    plt.scatter(x = entropy_region_sizes*replications, y = recalls, c="red", alpha=.4, s = 30, label="recall")
    plt.xlabel("number of neighbors")
    plt.xlabel('performance')
    plt.title("Performance vs. Entropy Region Size")
    plt.ylim((0,1))
    plt.legend()
    plt.grid()
    plt.show()

###########################################################
#Results:
# 1) percentage of peaks correctly identified drops off pretty fast with increasing densiy
# 2) lower entropy region size is the best, somewhere between 0 and 2
# 2.5) distances to closest peaks is optimized at 2, percentage of peaks found is optimized at 0
# 3) might even be better without smoothed entropy... test this
###########################################################

###########################################################
#################   MULTIVARIATE TESTS  ###################
###########################################################

###########################################################
# Test 6: Noise and alpha vs Perforance
###########################################################
def testNoiseAndAlpha():
    
    alphas = np.arange(0.9, .99, 0.1)
    noiselevels = np.arange(2, 100, 10)
    performances = []
    xNoise = []
    xAlpha = []
    for noise in noiselevels:
        data, peaks = generate(noise = noise)
        peaks.sort()
        for alpha in alphas:
            xNoise.append(noise)
            xAlpha.append(alpha)
            spectrum = Spectrum(data = data, alpha = alpha)
            spectrum.findPeaks(alpha = alpha)
            performances.append(spectrum.assess(spectrum.peaks, peaks))
###########################################################
# Test 7: Alpha and Neighborhood size vs performance 
###########################################################
def test_alpha_and_NS(noise = 10, data_size=1500, n_peaks = 10):
    """
    To do
    """
    #test on 3 datasets
    #change plots
    alphas = np.concatenate([np.linspace(0.7,0.92, 8), np.linspace(0.92, 0.99, 10)])
    pltNos = np.arange(331, 340, 1)
   
    data, peaks = generate(noise=noise, nPeaks=n_peaks)
    
  
    peaks.sort()
    spectrum = Spectrum(data = data)
    neighborhood_sizes = np.arange(3, 21, 1)
    avg_performances = []

    for index, alpha in enumerate(alphas):

        performances = []
        recalls = []
        performance_by_NS = [[] for i in range(len(neighborhood_sizes))]
        recall_by_NS = [[] for i in range(len(neighborhood_sizes))]
        for j, neighborhood_size in enumerate(neighborhood_sizes):
            print("#"*150)
            print('alpha = ', alpha)
            print('neighborhoodsize = ', neighborhood_size)
            
            spectrum = Spectrum(data = data)
            spectrum.findPeaks(alpha=alpha, neighborhoodSize=neighborhood_size)
            spectrum.peaks.sort()
        

        
            performance , recall  = spectrum.assess(spectrum.peaks, peaks)
            
          
            performances.append(performance)
            performance_by_NS[j].append(performance)
            recalls.append(recall)
            recall_by_NS[j].append(recall)
        avg_performances.append(np.mean(performances))
        if not index % 2: #if even
            plt.subplot(pltNos[index//2])
            plt.ylim([0, 1])
    
            p = np.polyfit(x=neighborhood_sizes, y = performances, deg=2)
            plt.scatter(x = neighborhood_sizes, y = performances)
            plt.scatter(x = neighborhood_sizes, y = recalls, c = 'red')
            xfit = np.linspace(0, 20, 30)
            yfit = np.polyval(p, xfit)
            plt.plot(xfit, yfit, "--")
    
            plt.title(f"Alpha = {round(alpha, 2)}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.scatter(x = alphas, y = avg_performances)
    p = np.polyfit(x = alphas, y = avg_performances, deg = 2)
    xfit = np.linspace(0.7, 1, 30)
    yfit = np.polyval(p, xfit)

    plt.plot(xfit, yfit, '--')
    plt.title("Average Performance by alpha across neighborhood sizes")
    plt.xlabel('Alpha')
    plt.ylim((0,1))
    plt.ylabel('Performance')
    plt.show()

    performance_by_NS = [np.mean(p) for p in performance_by_NS]
    plt.scatter(x = neighborhood_sizes, y = performance_by_NS)
    p = np.polyfit(x = neighborhood_sizes, y = performance_by_NS, deg = 2)
    xfit = np.linspace(0, len(neighborhood_sizes), 30)
    yfit = np.polyval(p, xfit)

    plt.plot(xfit, yfit, '--')
    plt.title("Average Performance Across by neighborhood size across alpha levels")
    plt.xlabel('Neighborhood Size')
    plt.ylabel('Performance')
    plt.ylim((0,1))
    plt.show()
###########################################################
## OUTCOME:
# optimum neighborhood size is around 3 - 5
# Increasing neighborhood size results in a lower percentage of peaks found.


###########################################################

###########################################################
# Test 8: Noise and Neighborhood Size vs Performance
###########################################################
def test_alpha_and_noise(data_size = 1500, neighborhood_size=5):
    """
    To do
    """
    data_size = 1500
    plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey=True)
    plt.title("Algorithm performance vs Noise with varying outlier sensitivity")
    alphas = np.concatenate([np.linspace(0.7,0.92, 8), np.linspace(0.92, 0.99, 10)])
    pltNos = np.arange(331, 340, 1)
    avg_performance = []

    for index, alpha in enumerate(alphas):

        ranges = range(2, 30, 1)
        datasets = [generate(n=data_size, nPeaks = 10, noise= i) for i in ranges]
        performances = []
        noiselevel = [i for i in ranges]

        for data, peaks in datasets:
       
            peaks.sort()
            spectrum = Spectrum(data = data)
            spectrum.findPeaks(alpha=alpha, neighborhoodSize=neighborhood_size)
            print("peaks found:", spectrum.peaks)
            #performances.append(spectrum.bootstrap(spectrum.peaks, peaks, data_size, sample_size=150))
            performances.append(spectrum.assess(spectrum.peaks, peaks, data_size))
        avg_performance.append(np.mean(performances))
        if not index % 2: #if even
            plt.subplot(pltNos[index//2])
        #plt.ylim([0, 1])
            plt.scatter(x = noiselevel, y = performances)
            plt.axhline(np.mean(performances), alpha=0.6)
            plt.ylim((0,1))
            plt.title(f"Alpha = {round(alpha, 2)}")
   
    plt.show()
    plt.scatter(x = alphas, y = avg_performance)
    p = np.polyfit(x = alphas, y = avg_performance, deg = 2)
    xfit = np.linspace(0.7, 1, 30)
    yfit = np.polyval(p, xfit)

    plt.plot(xfit, yfit, '--')
    plt.title("Average Performance Across all noise levels")
    plt.xlabel('Alpha')
    plt.ylim((0,1))
    plt.ylabel('Performance')
    
    plt.show()

################################################################################################################
# Outcome:
# When averaged across all noise levels, an alpha of 0.94-0.95 performs best. This pattern seems to stay consistent for vaying noise levels. At very high noise (30% of average peak height), performance began to drop, but less so for alphas at 0.94 and 0.95
################################################################################################################
#test_entropy_region_size()
if __name__=='__main__':
    test_time()   
#test_alpha_and_NS(data_size = 2000, n_peaks=20)