import numpy as np
from scipy import optimize

# The methods implemented in scipy are meant to find the minima,
# thus to find the maxima we have to negate the functions
f_to_optimize = lambda x: -np.sin(x)
optimize.fmin_cg(f_to_optimize, x0=[0])