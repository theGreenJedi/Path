import pandas as pd
import sys
import numpy as np
from datetime import datetime as dt
from matplotlib import finance
 
to_float = lambda x: .1 * float(x.strip() or np.nan)
to_date = lambda x: dt.strptime(x, "%Y%m%d")
cols = [4, 11, 25]
conv_dict = dict( (col, to_float) for col in cols) 
 
conv_dict[1] = to_date
cols.append(1)
 
headers = ['dates', 'avg_ws', 'avg_temp', 'avg_pres']
df = pd.read_csv(sys.argv[1], usecols=cols, names=headers, index_col=[0], converters=conv_dict)
 
#EWN start Mar 22, 1996
start = dt(1996, 3, 22)
end = dt(2013, 5, 4)
 
symbol = "EWN"
quotes = finance.quotes_historical_yahoo(symbol, start, end, asobject=True)
 
# Create date time index
dt_idx = pd.DatetimeIndex(quotes.date)
 
#Create data frame
df2 = pd.DataFrame(quotes.close, index=dt_idx, columns=[symbol])
print df2.head()
 
df3 = df.join(df2)
 
print df3.corr()
print
