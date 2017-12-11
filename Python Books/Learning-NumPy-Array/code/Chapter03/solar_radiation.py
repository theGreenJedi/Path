import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt
 
to_float = lambda x: float(x.strip() or np.nan)
to_year = lambda x: dt.strptime(x, "%Y%m%d").year
 
years, avg_temp, Q = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11, 20), unpack=True, converters={1: to_year, 20: to_float})
ma
# Measurements are in .1 degrees Celcius
avg_temp = .1 * avg_temp
 
Q = ma.masked_invalid(Q)
print "# temperature values", len(avg_temp), "# radiation values", len(Q.compressed())
print "Radiation Min", Q.min(), "Radiation Max", Q.max()
print "Radiation Average", Q.compressed().mean(), "Std Dev", Q.std()
 
match_temp =  avg_temp[np.logical_not(np.isnan(Q))]
print "Correlation Coefficient", np.corrcoef(match_temp, Q.compressed())[0][1]
 
avg_temps = []
avg_qs = []
year_range = range(int(years[0]), int(years[-1]) - 1)
 
for year in year_range:
   indices = np.where(years == year)
   avg_temps.append(avg_temp[indices].mean())
   avg_qs.append(Q[indices].mean())
 
def percents(a):
   return 100 * np.diff(a)/a[:-1]
 
plt.subplot(211)
plt.title("Global Radiation Histogram")
plt.hist(Q.compressed(), 200)
 
plt.subplot(212)
plt.title("Changes in Average Yearly Temperature &amp; Radiation")
plt.plot(year_range[1:], percents(avg_temps), label='% Change Temperature')
plt.plot(year_range[1:], percents(avg_qs), label='% Change Radiation')
plt.legend(prop={'size':'x-small'})
plt.show()
