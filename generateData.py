import numpy as np, matplotlib.pyplot as plt, pandas as pd


def generate( n:int=1000, nPeaks:int = 10, noise:float = 1, peakwidthsVariability:float = 1, peakHeightVariability:float = 20) -> None:
    """
    
    """
    peakCenters = np.random.randint(low=20, high=n-20,size=nPeaks)
    minimumPeakWidth = 10-peakwidthsVariability
    maximumPeakWidth = 10+peakwidthsVariability
    peakwidths = np.random.randint(low=minimumPeakWidth, high=maximumPeakWidth, size=nPeaks)
    peakHeights = [np.random.rand() * peakHeightVariability + 90 for i in range(nPeaks)]

    #an initial string of random numbers centered on 0:
    data = np.random.normal(location = 0, scale = noise, size =n)
    
    #Make the peaks and add them to the data:
    for center, width, height in zip(peakCenters, peakwidths, peakHeights):
        peak = makePeak(width=width, height=height)
        for index1, index2 in enumerate(range(center-width, center+width+1)):
            data[index2] -= peak[index1]
            
    data = pd.DataFrame(zip([i for i in range(len(data))], data))
    return [data, peakCenters]

def makePeak(width:int, height:int=100):
    """
    
    """
   
    rightSide = [-x**2 for x in range(width+1)]
    leftSide = [x for x in rightSide[:0:-1]]

    peak = leftSide+rightSide
    scale = -height/leftSide[0]
    peak = [(x*scale)+height for x in peak]
    
    return peak



if __name__ == '__main__':
    data = generate(noise=10)
    plt.plot(data[0])
    #[plt.axvline(x=peak, c="red") for peak in data[1]]
    plt.show()
