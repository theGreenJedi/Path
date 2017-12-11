import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt
import calendar as cal
 
to_float = lambda x: float(x.strip() or np.nan)
to_month = lambda x: dt.strptime(x, "%Y%m%d").month
months, duration, rain = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 21, 22), unpack=True, converters={1: to_month, 21: to_float, 22: to_float})
 
# Remove -1 values
rain[rain == -1] = 0
 
# Measurements are in .1 mm 
rain = .1 * ma.masked_invalid(rain)
 
# Measurements are in .1 hours 
duration = .1 * ma.masked_invalid(duration)
 
print "# Rain values", len(rain.compressed())
print "Min Rain hours ", rain.min(), "Max Rain hours", rain.max()
print "Average", rain.mean(), "Std. Dev", rain.std()
 
mask = ~duration.mask & ~rain.mask
print "Correlation with duration", np.corrcoef(duration[mask], rain[mask])[0][1]

