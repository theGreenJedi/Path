import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import leastsq
 
to_dayofyear = lambda x: dt.strptime(x, "%Y%m%d").timetuple().tm_yday
days, temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_dayofyear})
temp = .1 * temp
 
def model(p, d):
   a, b, w, c = p
   return a + b * np.cos(w * d + c)
 
def error(p, d, t):
   return t - model(p, d)
 
p0 = [.1, 1, .01, .01]
params = leastsq(error, p0, args=(days, temp))[0]
print params
rng = np.arange(1, 366)
avgs = np.zeros(366)
 
for i in rng:
   indices = np.where(days == i)
   avgs[i-1] = temp[indices].mean()
 
plt.plot(avgs)
plt.plot(model(params, rng))
plt.show()
