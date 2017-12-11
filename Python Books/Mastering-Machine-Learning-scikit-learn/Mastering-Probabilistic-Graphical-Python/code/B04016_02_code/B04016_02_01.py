# Firstly we need to import Factor
from pgmpy.factors import Factor
# Each factor is represented by its scope,
# cardinality of each variable in the scope and their values
phi = Factor(['A', 'B'], [2, 2], [1000, 1, 5, 100])
