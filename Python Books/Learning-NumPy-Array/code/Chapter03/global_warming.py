import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt
 
to_year = lambda x: dt.strptime(x, "%Y%m%d").year
 
years, avg_temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_year})
 
# Measurements are in .1 degrees Celcius
avg_temp = .1 * avg_temp
 
N = len(avg_temp)
print "First Year", years[0], "Last Year", years[-1]
assert N == len(years)
assert years[:N/2].mean() < years[N/2:].mean()
print "First half average", avg_temp[:N/2].mean(), "Std Dev", avg_temp[:N/2].std()
print "Second half average", avg_temp[N/2:].mean(), "Std Dev", avg_temp[N/2:].std()
 
avgs = []
year_range = range(int(years[0]), int(years[-1]) - 1)
 
for year in year_range:
   indices = np.where(years == year)
   avgs.append(avg_temp[indices].mean())
 
plt.plot(year_range, avgs, 'r-', label="Yearly Averages")
plt.plot(year_range, np.ones(len(avgs)) * np.mean(avgs))
plt.legend(prop={'size':'x-small'})
plt.show()
