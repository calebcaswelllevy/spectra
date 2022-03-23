
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import string

def parse(file_path:string):
    
    with open(file_path) as f:
        spectrum = [line.strip('\n').split(' ') for line in f]
        df = pd.DataFrame(list(zip([float(line[0]) for line in spectrum], [float(line[1]) for line in spectrum])), columns=["Wavelength", "Amplitude"]) 
    
    return df    

if __name__ == '__main__':    
    df = parse('./data/lte08000-4.50-0.0_0.30-2.50micron.dat')
    plt.plot(df['Amplitude'])
    plt.show()
