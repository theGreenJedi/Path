import numpy as np
import sys

to_float = lambda x: float(x.strip() or np.nan)

#Measurements are in tenths of degrees
min_temp, max_temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(12, 14), unpack=True, converters={12: to_float, 14: to_float}) * .1
print "# Records", len(min_temp), len(max_temp)
print "Minimum", np.nanmin(min_temp)
print "Maximum", np.nanmax(max_temp)
