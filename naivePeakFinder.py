import numpy as np, pandas as pd, matplotlib.pyplot as plt

def findPeaks(wavelengths:pd.Series):
    """
    """
    peaks = []
    for wavelength, index in enumerate(wavelengths):
        left, right = index -1, index + 1
        if index == 0 and wavelength < wavelengths[right]:
            peaks.append(index)
        elif index == len(wavelengths) and wavelength[left] > wavelength:
            peaks.append(index)
        elif wavelength < wavelengths[left] and wavelength < wavelengths[right]:
            peaks.append(index)
    return peaks

def graphPeaks(peaks:list, spectrum:pd.DataFrame):
    plt.plot(spectrum[0,0], spectrum[0,1])


x = [d for d in range(100)]
