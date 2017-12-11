phi1 = Factor(['a', 'b'], [2, 2], [1000, 1, 5, 100])
phi2 = Factor(['b', 'c'], [2, 3], [1, 100, 5, 200, 3, 1000])
# Factors product can be accomplished with the * (product)
# operator
phi = phi1 * phi2
phi.scope()
print(phi)
# or with product method
phi_new = phi.product(phi1, phi2)
# would produce a factor with phi_new = phi * phi1 * phi2
