import parseData, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft

def isPowerOfTwo(x:int):
    """
    Uses bitwise and to check if number is power of two
    """
    print('length is now: ', x)
    return (x != 0) and (x & (x-1)) == 0
def makePowerOfTwo(x:pd.Series):
    if not (isPowerOfTwo(len(x))):
        """
        print("Warning, length of input not power of two!")
        n = 0
        while not isPowerOfTwo(len(x)):
            n += 1
            x = x.drop(index=len(x)-1)
        print(f"Dropped {n} values to make it a power of two...")
        """
        powerOfTwo = 2
        while True:
          
            if powerOfTwo*2 > len(x):
                print(f"closest power of two to {len(x)} is {powerOfTwo}")
                return x[0:powerOfTwo]
            else:
                powerOfTwo *= 2
    return x

def FFT(x:pd.Series):
    """
    A recursive implementation of the 1D Cooley-Tukey FFT.
    Input should have length of of power of two.
    """
    N = len(x)
    if N == 1:
        return x

    else:
        x_even = FFT(x[::2])
        x_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)

        print("x_even: ", x_even)
        print("factor: ", factor)
        print("x_odd:", x_odd)
        print("++++++++++++++++++\n")
        X = np.concatenate(\
            [x_even+factor*x_odd,
             x_even+factor*x_odd])
        return X

if __name__ == '__main__':
    df = parseData.parse('./data/lte08000-4.50-0.0_0.30-2.50micron.dat')
    plt.style.use('seaborn-poster')
    print(df.head())
    
    # sampling rate
    sr = 256
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,1,ts)

    #prepare data:
    x = makePowerOfTwo(df['Amplitude'])
    print("length of x = ", len(x))
    
    N = 600
    T = 1.0 / 800.0
    
    yf = rfft(list(x))
    lowFreq = yf.copy()
    lowFreq2 = yf.copy()
    #calc frequency:
    
    xf = rfftfreq(N, T)[:N//2]

    with PdfPages(r'fft.pdf') as export_pdf:
        plt.bar(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.title("Results of FFT:")
        export_pdf.savefig()
        plt.close()
    
    """
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.show()"""
    
    
    

    yf[0:39]= 0
    """plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.show()"""
    zero = [0 for number in range(len(x))]
    filtered = irfft(yf)



    with PdfPages(r'output.pdf') as export_pdf:

        plt.plot(x, alpha=.6)
        plt.title('Original Spectrum with varying frequency bands')
        plt.plot(zero, color="k", linewidth=".5")
        lowFreq[6:] = 0
       
        plt.plot(irfft(lowFreq), color="red")
        
        lowFreq[5:] = 0
        plt.plot(irfft(lowFreq), color = "orange")

        lowFreq[4:] = 0
        plt.plot(irfft(lowFreq), color = "yellow")
        
        lowFreq[3:] = 0
        plt.plot(irfft(lowFreq), color = "blue")

        lowFreq[2:] = 0
        plt.plot(irfft(lowFreq), color = "purple")

        export_pdf.savefig()
        plt.show()
        plt.close()


        plt.plot(x, alpha=.6)
        plt.title('Individual components')
        plt.plot(zero, color="k", linewidth=".5")
        n = len(lowFreq2)
        six = [0,0,0,0,0,lowFreq2[6]] + [0 for number in range(n-6)]
        five = [0,0,0,0,lowFreq2[5]] + [0 for number in range(n-5)]
        four = [0,0,0,lowFreq2[4]] + [0 for number in range(n-4)]
        three = [0,0,lowFreq2[3]] + [0 for number in range(n-3)]
        two = [0,lowFreq2[2]] + [0 for number in range(n-2)]
        one = [lowFreq2[1]] + [0 for number in range(n-1)]

        plt.plot(irfft(one), color="red")
        

        plt.plot(irfft(two), color = "orange")


        plt.plot(irfft(three), color = "yellow")
        
    
        plt.plot(irfft(four), color = "blue")


        plt.plot(irfft(five), color = "indigo")

        plt.plot(irfft(five), color = "purple")

        export_pdf.savefig()
        plt.show()
        plt.close()
        
        
        

        yf[0:106]= 0
        """plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.show()"""

        filtered = irfft(yf)
        plt.plot(x, alpha=.6)
        plt.plot(filtered, alpha=.6)
        plt.plot(zero, color="k", linewidth=".5")
       
        plt.title(f"{(106/1048576)*100}% lowest frequencies removed")
        export_pdf.savefig()
        plt.close()


        yf[0:400] = 0

        filtered = irfft(yf)
        plt.plot(x, alpha=.6)
        plt.plot(filtered, alpha=.6)
        plt.title(f"{(400/1048576)*100}% lowest frequencies removed")
        plt.plot(zero, color="k", linewidth=".5")
        export_pdf.savefig()
        plt.close()

        yf[0:1000] = 0
        filtered = irfft(yf)
        plt.plot(x, alpha=.6)
        plt.plot(filtered, alpha=.6)
        plt.title(f"{(1000/1048576)*100}% lowest frequencies removed")
        plt.plot(zero, color="k", linewidth=".5")
        export_pdf.savefig()
        plt.close()

        yf[0:10000] = 0
        filtered = irfft(yf)
        plt.plot(x, alpha=.6)
        plt.plot(filtered, alpha=.6)
        plt.title(f"{(10000/1048576)*100}% lowest frequencies removed")
        plt.plot(zero, color="k", linewidth=".5")
        export_pdf.savefig()
        plt.close()

        yf[0:20000] = 0
        filtered = irfft(yf)
        plt.plot(x, alpha=.6)
        plt.plot(filtered, alpha=.6)
        plt.title(f"{(20000/1048576)*100}% lowest frequencies removed")
        plt.plot(zero, color="k", linewidth=".5")
        export_pdf.savefig()
        plt.close()

        yf[0:100000] = 0
        filtered = irfft(yf)
        plt.plot(x, alpha=.6)
        plt.plot(filtered, alpha=.6)
        plt.title(f"{(100000/1048576)*100}% lowest frequencies removed")
        plt.plot(zero, color="k", linewidth=".5")
        export_pdf.savefig()
        plt.close()

