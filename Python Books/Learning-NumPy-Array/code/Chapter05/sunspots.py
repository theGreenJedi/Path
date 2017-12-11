import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal

data = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1,), unpack=True, skiprows=1) 
#reverse order
data = data[::-1]

mins = signal.argrelmin(data)[0]
maxs = signal.argrelmax(data)[0]
extrema = np.concatenate((mins, maxs))

year_range = np.arange(1700, 1700 + len(data))

plt.plot(1700 + extrema, data[extrema], 'go')
plt.plot(year_range, data)
plt.show()
