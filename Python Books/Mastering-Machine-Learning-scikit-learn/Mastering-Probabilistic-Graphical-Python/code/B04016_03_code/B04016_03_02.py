# If we have some evidence for the network we can simply pass it
# as an argument to the query method in the form of
# {variable: state}
restaurant_inference.query(variables=['no_of_people'],
                           evidence={'location': 1})
restaurant_inference.query(variables=['no_of_people'],
                           evidence={'location': 1, 'quality': 1})
# In this case also we can simply pass the elimination order for
# the variables.
restaurant_inference.query(variables=['no_of_people'],
                           evidence={'location': 1},
                           elimination_order=['quality', 'cost'])
