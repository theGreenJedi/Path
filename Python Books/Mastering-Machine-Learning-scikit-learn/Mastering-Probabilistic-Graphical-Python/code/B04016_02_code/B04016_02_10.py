# We can also add factors into the model
from pgmpy.factors import Factor
import numpy as np
phi1 = Factor(['A', 'B'], [2, 2], np.random.rand(4))
phi2 = Factor(['B', 'C'], [2, 2], np.random.rand(4))
phi3 = Factor(['C', 'A'], [2, 2], np.random.rand(4))
factor_graph.add_factors(phi1, phi2, phi3)
