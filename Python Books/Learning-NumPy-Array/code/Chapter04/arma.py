import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import leastsq
 
temp = .1 * np.loadtxt(sys.argv[1], delimiter=',', usecols=(11,), unpack=True)
cutoff = 0.9 * len(temp)
 
def model(p, ma1):
   c0, c1 = p
 
   return c0 + c1 * ma1
 
def error(p, t, ma1):
   return t - model(p, ma1)
 
p0 = [.1, .1]
 
def ar(a):
   ar_p = [1.06517683, -0.08293789]
 
   return ar_p[0] * a[1:-1] + ar_p[1] * a[:-2]
 
err_terms = temp[2:cutoff] - ar(temp[:cutoff])
params = leastsq(error, p0, args=(err_terms[1:], err_terms[:-1]))[0]
print params
 
err_terms = temp[cutoff+1:] - ar(temp[cutoff-1:])
delta = np.abs(error(params, err_terms[1:], err_terms[:-1]))
print "% delta less than 2", (100. * len(delta[delta <= 2]))/len(delta)
 
plt.hist(delta, bins = 10, normed = True)
plt.show()
