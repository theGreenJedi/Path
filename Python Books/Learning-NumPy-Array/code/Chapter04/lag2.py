import sys
import numpy as np
import matplotlib.pyplot as plt
 
temp = .1 * np.loadtxt(sys.argv[1], delimiter=',', usecols=(11,), unpack=True)
cutoff = 0.9 * len(temp)
A = np.zeros((2, cutoff - 2), float)
 
A[0, ] = temp[:cutoff - 2]
A[1, ] = temp[1 :cutoff - 1]
 
b = temp[2 : cutoff]
(x, residuals, rank, s) = np.linalg.lstsq(A.T, b)
print x
fit = x[0] * temp[cutoff-1:-2] + x[1] * temp[cutoff:-1]
delta = np.abs(temp[cutoff + 1:] - fit)
plt.hist(delta, bins = 10, normed=True)
plt.show()
