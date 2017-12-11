import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

data = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1,), unpack=True, skiprows=1) 
#reverse order
data = data[::-1]

mins = signal.argrelmin(data)[0]
maxs = signal.argrelmax(data)[0]
extrema = np.concatenate((mins, maxs))

year_range = np.arange(1700, 1700 + len(data))
spl_min = interpolate.interp1d(mins, data[mins], kind='cubic')
min_rng = np.arange(mins.min(), mins.max())
l_env = spl_min(min_rng)

spl_max = interpolate.interp1d(maxs, data[maxs], kind='cubic')
max_rng = np.arange(maxs.min(), maxs.max())
u_env = spl_max(max_rng)

inclusive_rng = np.arange(max(min_rng[0], max_rng[0]), min(min_rng[-1], max_rng[-1]))
mid = (spl_max(inclusive_rng) + spl_min(inclusive_rng))/2

plt.plot(year_range, data)
plt.plot(1700 + min_rng, l_env, '-x')
plt.plot(1700 + max_rng, u_env, '-x')
plt.plot(1700 + inclusive_rng, mid, '--')
plt.show()
