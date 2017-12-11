import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import leastsq
 
to_dayofyear = lambda x: dt.strptime(x, "%Y%m%d").timetuple().tm_yday
days, temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_dayofyear})
temp = .1 * temp
cutoff = 0.9 * len(temp)
 
def error(p, d, t, lag2, lag1):
   l2, l1, d2, d1, d0 = p
 
   return t - l2 * lag2 + l1 * lag1 + d2 * d ** 2 + d1 * d + d0
 
p0 = [-0.08293789,  1.06517683, -4.91072584e-04,   1.92682505e-01,  -3.97182941e+00]
params = leastsq(error, p0, args=(days[2:cutoff], temp[2:cutoff], temp[:cutoff - 2], temp[1 :cutoff - 1]))[0]
print params
delta = np.abs(error(params, days[cutoff+1:], temp[cutoff+1:], temp[cutoff-1:-2], temp[cutoff:-1]))
 
plt.hist(delta, bins = 10, normed = True)
plt.show()
