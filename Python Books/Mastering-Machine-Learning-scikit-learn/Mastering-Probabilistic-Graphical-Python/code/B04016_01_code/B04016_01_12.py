from pgmpy.factors import TabularCPD
# For creating a TabularCPD object we need to pass three
# arguments: the variable name, its cardinality that is the number
# of states of the random variable and the probability value
# corresponding each state.
quality = TabularCPD(variable='Quality',
                     variable_card=3,
                     values=[[0.3], [0.5], [0.2]])
print(quality)
quality.variables
quality.cardinality
quality.values