import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt
import calendar as cal
 
to_float = lambda x: float(x.strip() or np.nan)
to_month = lambda x: dt.strptime(x, "%Y%m%d").month
months, sun_hours, rain_hours = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 18, 21), unpack=True, converters={1: to_month, 18: to_float, 21: to_float})
 
# Measurements are in .1 hours 
rain_hours = .1 * ma.masked_invalid(rain_hours)
 
#Get rid of -1 values
print "# -1 values Before", len(sun_hours[sun_hours == -1])
sun_hours[sun_hours == -1] = 0
print "# -1 values After", len(sun_hours[sun_hours == -1])
sun_hours = .1 * ma.masked_invalid(sun_hours)
 
print "# Rain hours values", len(rain_hours.compressed())
print "Min Rain hours ", rain_hours.min(), "Max Rain hours", rain_hours.max()
print "Average", rain_hours.mean(), "Std. Dev", rain_hours.std()
 
monthly_rain = []
monthly_sun = []
month_range = np.arange(int(months.min()), int(months.max()))
 
for month in month_range:
   indices = np.where(month == months)
   monthly_rain.append(rain_hours[indices].mean())
   monthly_sun.append(sun_hours[indices].mean())
 
plt.subplot(211)
plt.title("Precipitation Duration Histogram")
plt.hist(rain_hours[rain_hours > 0].compressed(), 200)
 
width = 0.42
ax = plt.subplot(212)
plt.title("Monthly Precipitation Duration")
plt.bar(month_range, monthly_rain, width, label='Rain Hours')
plt.bar(month_range + width, monthly_sun, width, color='red', label='Sun Hours')
plt.legend()
ax.set_xticklabels(cal.month_abbr[::2])
ax.set_ylabel('Hours')
plt.show()
