import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.signal

data = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1,), unpack=True, skiprows=1) 
#reverse order
data = data[::-1]

# Design the filter
b,a = scipy.signal.iirdesign(wp=0.2, ws=0.1, gstop=60, gpass=1, ftype='butter')

# Filter
filtered = scipy.signal.lfilter(b, a, data)

year_range = np.arange(1700, 1700 + len(data))
plt.plot(year_range, filtered, '--', label='Filtered')
plt.plot(year_range, data, label='Data')
plt.legend()
plt.show()
