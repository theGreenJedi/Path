import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt
import calendar as cal
 
 
to_float = lambda x: 0.1 * float(x.strip() or np.nan)
to_month = lambda x: dt.strptime(x, "%Y%m%d").month
months, avg_p, max_p, min_p = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 25, 26, 28), unpack=True, converters={1: to_month, 25: to_float, 26: to_float, 28: to_float})
 
max_p = ma.masked_invalid(max_p)
print "Maximum Pressure", max_p.max()
 
avg_p = ma.masked_invalid(avg_p)
print "Average Pressure", avg_p.mean(), "Std Dev", avg_p.std()
 
min_p = ma.masked_invalid(min_p)
print "Minimum Pressure", min_p.max()
 
monthly_pressure = []
maxes = []
mins = []
month_range = np.arange(int(months.min()), int(months.max()))
 
for month in month_range:
   indices = np.where(month == months)
   monthly_pressure.append(avg_p[indices].mean())
   maxes.append(max_p[indices].max())
   mins.append(min_p[indices].min())
 
plt.subplot(211)
plt.title("Pressure Histogram")
a, bins, b = plt.hist(avg_p.compressed(), 200, normed=True)
stdev = avg_p.std()
avg = avg_p.mean()
plt.plot(bins, 1/(stdev * np.sqrt(2 * np.pi)) * np.exp(- (bins - avg)**2/(2 * stdev**2)), 'r-')
 
ax = plt.subplot(212)
plt.title("Monthly Pressure")
plt.plot(month_range, monthly_pressure, 'bo', label="Average")
plt.plot(month_range, maxes, 'r^', label="Maximum Values")
plt.plot(month_range, mins, 'g>', label="Minumum Values")
ax.set_xticklabels(cal.month_abbr[::2])
plt.legend(prop={'size':'x-small'}, loc='best')
ax.set_ylabel('hPa')
plt.show()
