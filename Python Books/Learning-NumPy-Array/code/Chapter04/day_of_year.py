import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
 
to_dayofyear = lambda x: dt.strptime(x, "%Y%m%d").timetuple().tm_yday
days, temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_dayofyear})
temp = .1 * temp
cutoff = 0.9 * len(temp)
 
poly = np.polyfit(days[:cutoff], temp[:cutoff], 2)
print poly
 
delta = np.abs(np.polyval(poly, days[cutoff:]) - temp[cutoff:])
 
plt.hist(delta, bins = 10, normed = True)
plt.show()
