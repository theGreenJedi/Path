from pgmpy.models import MarkovModel
from pgmpy.factors import Factor
model = MarkovModel()
# Fig 2.7(a) represents the Markov model
model.add_nodes_from(['A', 'B', 'C', 'D'])
model.add_edges_from([('A', 'B'), ('B', 'C'),
                      ('C', 'D'), ('D', 'A')])
# Adding some factors.
phi_A_B = Factor(['A', 'B'], [2, 2], [1, 100,
phi_B_C = Factor(['B', 'C'], [2, 2], [100, 1,
phi_C_D = Factor(['C', 'D'], [2, 2], [1, 100,
phi_D_A = Factor(['D', 'A'], [2, 2], [100, 1,
model.add_factors(phi_A_B, phi_B_C, phi_C_D,
bayesian_model = model.to_bayesian_model()
bayesian_model.edges()
