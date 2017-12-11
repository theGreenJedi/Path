import pandas as pd
import sys
import numpy as np
from datetime import datetime as dt
 
to_float = lambda x: .1 * float(x.strip() or np.nan)
to_date = lambda x: dt.strptime(x, "%Y%m%d")
cols = [4, 11, 25]
conv_dict = dict( (col, to_float) for col in cols) 
 
conv_dict[1] = to_date
cols.append(1)
 
headers = ['dates', 'avg_ws', 'avg_temp', 'avg_pres']
df = pd.read_csv(sys.argv[1], usecols=cols, names=headers, index_col=[0], converters=conv_dict)
print df.head()
print
 
print df.tail()
print
 
print df.describe()
print
 
print df.corr()
print
