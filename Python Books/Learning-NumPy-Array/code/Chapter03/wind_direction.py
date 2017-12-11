import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
 
to_float = lambda x: float(x.strip() or np.nan)
wind_direction, avg_temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(2, 11), unpack=True, converters={2: to_float})
wind_direction = ma.masked_invalid(wind_direction)
 
# Measurements are in .1 degrees Celcius
avg_temp = .1 * avg_temp
 
avgs = []
 
for direction in xrange(360):
   indices = np.where(direction == wind_direction)
   avgs.append(avg_temp[indices].mean())
 
plt.subplot(211)
plt.title("Wind Direction Histogram")
plt.hist(wind_direction.compressed(), 200)
 
plt.subplot(212)
plt.title("Average Temperature vs Wind Direction")
plt.plot(np.arange(360), avgs)
plt.show()
