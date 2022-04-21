from Palshikar2009Peak import Spectrum
from generateData import make_group
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import numpy as np

labels = ["A"]*10 + ["B"]*10 + ["C"]*10 + ["D"] * 10 + ["E"]*10 + ["F"] * 10
data = []

[data.extend(make_group()) for i in range(6)]

spectra = []

[spectra.append(Spectrum(data=values)) for values, realpeaks in data]
[spectrum.findPeaks() for spectrum in spectra]

peakset = []
[peakset.append(index)  for spectrum in spectra for index in spectrum.getPeaks() if index not in peakset]
print("*"*100)
print("peakset = ", peakset)

X = [[] for i in range(len(spectra))]
for spectrum, peak_holder in zip(spectra, X):
    for peak in peakset:
        peak_holder.append(spectrum.getData()[1][peak])
X = np.array(X)
X = StandardScaler().fit_transform(X)


print('-'*100)
print(X)
clustering = OPTICS(min_samples=4).fit(X)
print("<>"*100)
[print(f"{label} : {cluster}") for label, cluster in zip(labels, clustering.labels_)] 


################
##IDEAS
##with an iterative approach, it should be possible to find peaks on non-rectified spectra
##detrend by subtraction
##measure height by finding the point where the tip of verticle line is no longer equidistant from the spectrum on either side
#1) make a spectrum that has a hump
#   - multiply by a hump-shaped function
#2) experiment with detrending it
#   - something simple or it might not be worth it
#3) make a version of the algorithm that finds LOCAL KDE 
#4) see how it performs on detrended and non-detrended spectra