import numpy as np
import sys
import numpy.ma as ma
from datetime import datetime as dt
 
to_float = lambda x: float(x.strip() or np.nan)
to_date = lambda x: dt.strptime(x, "%Y%m%d").toordinal()
 
dates, avg_temp, min_temp, max_temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11, 12, 14), unpack=True, converters={1: to_date, 12: to_float, 14: to_float})
 
# Measurements are in .1 degrees Celcius
avg_temp = .1 * avg_temp
min_temp = .1 * min_temp
max_temp = .1 * max_temp
 
#Freezing %
print "% days min below 0", 100 * len(min_temp[min_temp < 0])/float(len(min_temp))
print "% days max below 0", 100 * len(max_temp[max_temp < 0])/float(len(max_temp))
print
 
#Daily ranges
ranges = max_temp - min_temp
print "Minimum daily range", np.nanmin(ranges)
print "Maximum daily range", np.nanmax(ranges)
 
masked_ranges = ma.array(ranges, mask = np.isnan(ranges))
print "Average daily range", masked_ranges.mean()
print "Standard deviation", masked_ranges.std()
 
masked_mins = ma.array(min_temp, mask = np.isnan(min_temp))
print "Average minimum temperature", masked_mins.mean(), "Standard deviation", masked_mins.std()
 
masked_maxs = ma.array(max_temp, mask = np.isnan(max_temp))
print "Average maximum temperature", masked_maxs.mean(), "Standard deviation", masked_maxs.std()
