from __future__ import print_function
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM

# Downloading the data
date1 = datetime.date(1995, 1, 1)
date2 = datetime.date(2012, 1, 6)

# start date
# end date
# get quotes from yahoo finance
quotes = quotes_historical_yahoo("INTC", date1, date2)

# unpack
quotes
dates = np.array([q[0] for q in quotes], dtype=int)
close_v = np.array([q[2] for q in quotes])
volume = np.array([q[5] for q in quotes])[1:]

# take diff of close value
# this makes len(diff) = len(close_t) - 1
# therefore, others quantity also need to be shifted
diff = close_v[1:] - close_v[:-1]
dates = dates[1:]
close_v = close_v[1:]

# pack diff and volume for training
X = np.column_stack([diff, volume])

# Run Gaussian HMM
n_components = 5

# make an HMM instance and execute fit
model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)
model.fit([X])

# predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

# print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))

years = YearLocator()

# every year
months = MonthLocator() # every month
yearsFmt = DateFormatter('%Y')
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(n_components):
# use fancy indexing to plot data in each state
    idx = (hidden_states == i)
    ax.plot_date(dates[idx], close_v[idx], 'o',
    label="%dth hidden state" % i)
ax.legend()

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()

# format the coords message box
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)
ax.set_xlabel('Year')
ax.set_ylabel('Closing Volume')
fig.autofmt_xdate()
plt.show()
