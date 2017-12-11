import numpy as np
import pandas as pd
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
 
to_date = lambda x: dt.strptime(x, "%Y%m%d").toordinal()
 
dates, avg_temp = np.loadtxt(sys.argv[1], delimiter=',', usecols=(1, 11), unpack=True, converters={1: to_date})
dtidx = pd.DatetimeIndex([dt.fromordinal(int(date)) for date in dates])
data = pd.Series(avg_temp * .1, index=dtidx)
 
fig = plt.figure()
fig.add_subplot(211)
lag_plot(data)
 
plt.figure()
autocorrelation_plot(data)
 
plt.figure()
resampled = data.resample('A')
resampled.plot()
plt.show()
