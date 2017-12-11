# In the preceding example phi, let's try to reduce to the context
# of b_0
phi = Factor(['a', 'b'], [2, 2], [1000, 1, 5, 100])
phi_reduced = phi.reduce(('b', 0), inplace=False)
print(phi_reduced)
phi_reduced.scope()
# If inplace=True (default), it would modify the original factor
# instead of returning a new object.
phi.reduce(('a', 1))
print(phi)
phi.scope()
# A factor can be also reduced with respect to more than one
# random variable
price_reduced = price.reduce([('quality', 0), ('location', 1)],
                             inplace=False)
price_reduced.scope()