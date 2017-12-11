import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import leastsq
 
to_ordinal = lambda x: dt.strptime(x, "%Y%m%d").toordinal()
ordinals, temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_ordinal})
days = np.array([dt.fromordinal(int(d)).timetuple().tm_yday for d in ordinals])
years = np.array([dt.fromordinal(int(d)).year for d in ordinals])
temp = .1 * temp
cutoff = 0.9 * len(temp)
 
avgs = np.zeros(366)
 
for i in xrange(1, 366):
   indices = np.where(days[:cutoff] == i)
   avgs[i-1] = temp[indices].mean()
 
def subtract_avgs(a, doy):
   return a - avgs[doy.astype(int)-1]
 
def subtract_trend(a, poly, b):
   return a - poly[0] * b - poly[1]
 
def print_stats(a):
   print "Min", a.min(), "Max", a.max(), "Mean", a.mean(), "Std", a.std()
   print
 
# Step 1. DOY avgs
less_avgs = subtract_avgs(temp[:cutoff], days[:cutoff])
print "After Subtracting DOY avgs"
print_stats(less_avgs)
 
# Step 2. Linear trend
trend = np.polyfit(years[:cutoff], less_avgs, 1)
print "Trend coeff", trend
less_trend = subtract_trend(less_avgs, trend, years[:cutoff])
print "After Subtracting Linear Trend"
print_stats(less_trend)
 
def model(p, lag2, lag1):
   l1, l2 = p
 
   return l2 * lag2 + l1 * lag1
 
def error(p, t, lag2, lag1):
   return t - model(p, lag2, lag1) 
 
p0 = [1.06517683, -0.08293789]
params = leastsq(error, p0, args=(less_trend[2:], less_trend[:-2], less_trend[1:-1]))[0]
print "AR params", params
 
#Step 1. again
less_avgs = subtract_avgs(temp[cutoff+1:], days[cutoff+1:])
 
#Step 2. again
less_trend = subtract_trend(less_avgs, trend, years[cutoff+1:])
 
delta = np.abs(error(params, less_trend[2:], less_trend[:-2], less_trend[1:-1]))
print "% delta less than 2", (100. * len(delta[delta <= 2]))/len(delta)
 
plt.hist(delta, bins = 10, normed = True)
plt.show()
