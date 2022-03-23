import numpy as np, pandas as pd, FourierTest, matplotlib.pyplot as plt, parseData

df = parseData.parse('./data/lte08000-4.50-0.0_0.30-2.50micron.dat')
print('Data read...')
print("running DFT...")
X = FourierTest.DFT(df)
print('DFT done...')
#Calculate the frequency
N = len(X)
n = np.arange(N)
T = N/100 #sampling rate
freq = n/T
plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.show()
