from pgmpy.models import MarkovModel
mm = MarkovModel()
mm.add_nodes_from(['A', 'B', 'C'])
mm.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
mm.add_factors(phi1, phi2, phi3)
factor_graph_from_mm = mm.to_factor_graph()
# While converting a markov model into factor graph, factor nodes
# would be automatically added the factor nodes would be in the
# form of phi_node1_node2_...
factor_graph_from_mm.nodes()
factor_graph.edges()
# FactorGraph to MarkovModel
phi = Factor(['A', 'B', 'C'], [2, 2, 2],
np.random.rand(8))
factor_graph = FactorGraph()
factor_graph.add_nodes_from(['A', 'B', 'C', 'phi'])
factor_graph.add_edges_from([('A', 'phi'), ('B', 'phi'), ('C', 'phi')])