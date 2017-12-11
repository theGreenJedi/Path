import numpy as np
from scipy import stats
import pandas

import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

df = sm.datasets.sunspots.load_pandas().data

df.index = pandas.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del df["YEAR"]

model = sm.tsa.ARMA(df, (2,1)).fit()

year_today = datetime.date.today().year

#Big Brother is watching you!
prediction = model.predict('1984', str(year_today), dynamic=True)

df.plot()
prediction.plot(style='--', label='Prediction');
plt.legend();
plt.show()

