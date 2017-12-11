from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors import TabularCPD
# Now first create the model.
restaurant = BayesianModel([('location', 'cost'),
                            ('quality', 'cost'),
                            ('cost', 'no_of_people'),
                            ('location', 'no_of_people')])
cpd_location = TabularCPD('location', 2, [[0.6, 0.4]])
cpd_quality = TabularCPD('quality', 3, [[0.3, 0.5, 0.2]])
cpd_cost = TabularCPD('cost', 2,
                      [[0.8, 0.6, 0.1, 0.6, 0.6, 0.05],
                       [0.2, 0.1, 0.9, 0.4, 0.4, 0.95]],
                      ['location', 'quality'], [2, 3])
cpd_no_of_people = TabularCPD('no_of_people', 2,
                              [[0.6, 0.8, 0.1, 0.6],
                               [0.4, 0.2, 0.9, 0.4]],
                              ['cost', 'location'], [2, 2])
restaurant.add_cpds(cpd_location, cpd_quality,
                    cpd_cost, cpd_no_of_people)
# Creating the inference object of the model
restaurant_inference = VariableElimination(restaurant)
# Doing simple queries over one or multiple variables.
restaurant_inference.query(variables=['location'])
restaurant_inference.query(variables=['location', 'no_of_people'])
# We can also specify the order in which the variables are to be
# eliminated. If not specified pgmpy automatically computes the
# best possible elimination order.
restaurant_inference.query(variables=['no_of_people'],
                           elimination_order=['location', 'cost', 'quality'])
