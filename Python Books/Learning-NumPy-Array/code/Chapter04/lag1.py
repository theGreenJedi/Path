import sys
import numpy as np
import matplotlib.pyplot as plt
 
avg_temp = .1 * np.loadtxt(sys.argv[1], delimiter=',', usecols=(11,), unpack=True)
cutoff = 0.9 * len(avg_temp)
 
fig = plt.figure()
 
for degree in xrange(1, 4):
   poly = np.polyfit( avg_temp[: cutoff - 1], avg_temp[1 : cutoff], degree)
   print poly
 
   fit = np.polyval(poly, avg_temp[cutoff:-1])
   delta = np.abs(avg_temp[cutoff + 1:] - fit)
 
   for i in xrange(1, 4):
      print "# % <", i , "degree delta", 100.0 * len(delta[delta < i])/len(delta)

