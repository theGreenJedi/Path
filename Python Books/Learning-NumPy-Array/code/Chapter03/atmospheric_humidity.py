import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt
import calendar as cal
 
 
to_float = lambda x: float(x.strip() or np.nan)
to_month = lambda x: dt.strptime(x, "%Y%m%d").month
months, avg_h, max_h, min_h = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 35, 36, 38), unpack=True, converters={1: to_month, 35: to_float, 36: to_float, 38: to_float})
 
max_h = ma.masked_invalid(max_h)
print "Maximum Humidity", max_h.max()
 
avg_h = ma.masked_invalid(avg_h)
print "Average Humidity", avg_h.mean(), "Std Dev", avg_h.std()
 
min_h = ma.masked_invalid(min_h)
print "Minimum Humidity", min_h.min()
 
monthly_humidity = []
maxes = []
mins = []
month_range = np.arange(int(months.min()), int(months.max()))
 
for month in month_range:
   indices = np.where(month == months)
   monthly_humidity.append(avg_h[indices].mean())
   maxes.append(max_h[indices].max())
   mins.append(min_h[indices].min())
 
plt.subplot(211)
plt.title("Humidity Histogram")
plt.hist(avg_h.compressed(), 200)
 
ax = plt.subplot(212)
plt.title("Monthly Humidity")
plt.plot(month_range, monthly_humidity, 'bo', label="Average")
plt.plot(month_range, maxes, 'r^', label="Maximum Values")
plt.plot(month_range, mins, 'g>', label="Minumum Values")
ax.set_xticklabels(cal.month_abbr[::2])
plt.legend(prop={'size':'x-small'}, loc='best')
ax.set_ylabel('%')
plt.show()
