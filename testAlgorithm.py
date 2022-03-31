from generateData import generate
from Palshikar2009Peak import Spectrum
import matplotlib.pyplot as plt
import numpy as np
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
def testTime():
    dataSizes = np.arange(100, 100000, 10000)
    times = []
    for n in dataSizes:
        print('*'*150)
        print("starting test on size", n)
        data, peaks = generate(n)
        t1 = perf_counter()
        Spectrum(data = data)
        t2 = perf_counter()
        times.append(t2-t1)
        print("Completed in ", t2-t1)


    plt.scatter(x=dataSizes, y = times)
    plt.title("Time (ns) vs. data size")
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
    datasets = [generate(n=2000, nPeaks = i) for i in range(10, 40, 5)]
    performances = []
    peakNo = [i for i in range(10, 30, 5)]
    for data, peaks in datasets:
        spectrum = Spectrum(data = data)
        performances.append( spectrum.asses(spectrum.peaks, peaks))
    plt.scatter(x = peakNo, y = performances)
    plt.title("Algorithm performance vs peak density")
    plt.show()

###########################################################
# Test 3: noise vs. performance
###########################################################
def testNoise():
    datasets = [generate(n=2000, nPeaks = 10, noise= i) for i in range(2, 100, 10)]
    performances = []
    noiselevel = [i for i in range(2, 100, 10)]
    for data, peaks in datasets:
        spectrum = Spectrum(data = data)
        performances.append(spectrum.asses(spectrum.peaks, peaks))
    plt.scatter(x = noiselevel, y = performances)
    plt.title("Performance vs. Noise Level")
    plt.show()

###########################################################
# Test 4: noise vs. performance
###########################################################
def testNoise():
    datasets = [generate(n=2000, nPeaks = 10, noise= i) for i in range(2, 100, 10)]
    performances = []
    noiselevel = [i for i in range(2, 100, 10)]
    for data, peaks in datasets:
        spectrum = Spectrum(data = data)
        performances.append(spectrum.asses(spectrum.peaks, peaks))
    plt.scatter(x = noiselevel, y = performances)
    plt.title("Performance vs. Noise Level")
    plt.show()

###########################################################
# Test 5: alpha level vs. performance
###########################################################
def testAlpha():
    data, peaks = generate(noise = 10)
    performances = []
    alphas = np.arange(0.9, .99, 0.1)
    for alpha in alphas:
        spectrum = Spectrum(data = data, alpha = alpha)
        performances.append(spectrum.asses(spectrum.peaks, peaks))
    plt.scatter(x = alphas, y = performances)
    plt.title("Performance vs. alpha value")
    plt.show()

###########################################################
# Test 5: neighborhood size vs performance
###########################################################
def testNeighborhoodSize():
    data, peaks = generate(noise = 10)
    neighborhood_sizes = np.arange(2,20, 1)
    performances = []
    alphas = np.arange(0.9, .99, 0.1)
    for size in neighborhood_sizes:
        spectrum = Spectrum(data = data, neighborhoodSize=size)
        performances.append(spectrum.asses(spectrum.peaks, peaks))
    plt.scatter(x = alphas, y = performances)
    plt.title("Performance vs. Neighborhood size")
    plt.show()

###########################################################
#################   MULTIVARIATE TESTS  ###################
###########################################################

###########################################################
# Test 6: Noise and alpha vs Perforance
###########################################################
def testNoiseAndAlpha():
    pass
###########################################################
# Test 7: Alpha and Neighborhood size vs performance 
###########################################################
def testAlphaAndNS():
    pass
###########################################################
# Test 8: Noise and Neighborhood Size vs Performance
###########################################################
def testNoiseAndNS():
    pass