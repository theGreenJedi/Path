import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
 
to_dayofyear = lambda x: dt.strptime(x, "%Y%m%d").timetuple().tm_yday
days, temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_dayofyear})
temp = .1 * temp
cutoff = 0.9 * len(temp)
rng = np.arange(1, 366)
avgs = np.zeros(365)
avgs2 = np.zeros(365)
 
for i in rng: 
   indices = np.where(days[:cutoff] == i)
   avgs[i-1] = temp[indices].mean()
   indices = np.where(days[cutoff+1:] == i)
   avgs2[i-1] = temp[indices].mean()
 
 
poly = np.polyfit(rng, avgs, 2)
print poly
 
plt.plot(avgs2)
plt.plot(np.polyval(poly, rng))
plt.show()
