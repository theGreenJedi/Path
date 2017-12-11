import numpy as np
import sys
import matplotlib.pyplot as plt

data = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1,), unpack=True, skiprows=1) 
#reverse order
data = data[::-1]

year_range = np.arange(1700, 1700 + len(data))

def sma(arr, n):
   weights = np.ones(n) / n

   return np.convolve(weights, arr)[n-1:-n+1]

sma11 = sma(data, 11)
sma22 = sma(data, 22)

plt.plot(year_range, data, label='Data')
plt.plot(year_range[10:], sma11, '-x', label='SMA 11')
plt.plot(year_range[21:], sma22, '--', label='SMA 22')
plt.legend()
plt.show()

