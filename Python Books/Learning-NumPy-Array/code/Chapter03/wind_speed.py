import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
 
to_float = lambda x: float(x.strip() or np.nan)
wind_direction, wind_speed, avg_temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(2, 4, 11), unpack=True, converters={2: to_float, 4: to_float})
wind_direction = ma.masked_invalid(wind_direction)
wind_speed = ma.masked_invalid(wind_speed)
 
# Measurements are in .1 m/s
wind_speed = .1 * wind_speed
 
# Measurements are in .1 degrees 
avg_temp = .1 * avg_temp
 
print "# Wind Speed values", len(wind_speed.compressed())
print "Min speed", wind_speed.min(), "Max speed", wind_speed.max()
print "Average", wind_speed.mean(), "Std. Dev", wind_speed.std()
 
print "Correlation of wind speed and temperature", np.corrcoef(avg_temp[~wind_speed.mask], wind_speed.compressed())[0][1]

