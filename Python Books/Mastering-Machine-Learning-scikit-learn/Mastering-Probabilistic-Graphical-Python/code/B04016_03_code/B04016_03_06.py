from pgmpy.factors import Factor
phi1 = Factor(['a', 'b'], [2, 3], range(6))
phi2 = Factor(['b'], [3], range(3))
psi = phi1 / phi2
print(psi)