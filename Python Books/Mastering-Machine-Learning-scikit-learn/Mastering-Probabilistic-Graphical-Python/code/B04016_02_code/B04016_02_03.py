# In the preceding example phi, let's try to marginalize it with
# respect to B
phi_marginalized = phi.marginalize('B', inplace=False)
phi_marginalized.scope()
# If inplace=True (default), it would modify the original factor
# instead of returning a new one
phi.marginalize('A')
print(phi)
phi.scope()
# A factor can be also marginalized with respect to more than one
# random variable
price = Factor(['price', 'quality', 'location'],
               [2, 2, 2], np.arange(8))
price_marginalized = price.marginalize(['quality', 'location'],
                                       inplace=False)
price_marginalized.scope()
print(price_marginalized)
