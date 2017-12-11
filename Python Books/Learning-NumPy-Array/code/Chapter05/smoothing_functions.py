import numpy as np
import sys
import matplotlib.pyplot as plt

def smooth(weights, arr):
   return np.convolve(weights/weights.sum(), arr)

data = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1,), unpack=True, skiprows=1) 
#reverse order
data = data[::-1]

#Select last 50 years
data = data[-50:]
year_range = np.arange(1963, 2013)
print len(data), len(year_range)

plt.plot(year_range, data, label="Data")
plt.plot(year_range, smooth(np.hanning(22), data)[21:], 'x', label='Hanning 22')
plt.plot(year_range, smooth(np.bartlett(22), data)[21:], 'o', label='Bartlett 22')
plt.plot(year_range, smooth(np.blackman(22), data)[21:], '--', label='Blackman 22')
plt.plot(year_range, smooth(np.hamming(22), data)[21:], '^', label='Hamming 22')
plt.plot(year_range, smooth(np.kaiser(22, 14), data)[21:], ':', label='Kaiser 22')
plt.legend()
plt.show()

