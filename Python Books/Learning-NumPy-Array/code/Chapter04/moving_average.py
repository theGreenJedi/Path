import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import leastsq
 
temp = .1 * np.loadtxt(sys.argv[1], delimiter=',', usecols=(11,), unpack=True)
cutoff = 0.9 * len(temp)
 
def model(p, ma1):
   return p * ma1
 
def error(p, t, ma1):
   return t - model(p, ma1)
 
p0 = [.9]
mu = temp[:cutoff].mean()
params = leastsq(error, p0, args=(temp[1:cutoff] - mu, temp[:cutoff-1] - mu))[0]
print params
 
delta = np.abs(error(params, temp[cutoff+1:] - mu, temp[cutoff:-1] - mu))
print "% delta less than 2", (100. * len(delta[delta <= 2]))/len(delta)
 
plt.hist(delta, bins = 10, normed = True)
plt.show()
