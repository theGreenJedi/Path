# First import FactorGraph class from pgmpy.models
from pgmpy.models import FactorGraph
factor_graph = FactorGraph()
# Add nodes (both variable nodes and factor nodes) to the model
# as we did in previous other models
factor_graph.add_nodes_from(['A', 'B', 'C', 'D', 'phi1', 'phi2', 'phi3'])
# Add edges between all variable nodes and factor nodes
factor_graph.add_edges_from([('A', 'phi1'), ('B', 'phi1'),
                             ('B', 'phi2'), ('C', 'phi2'),
                             ('C', 'phi3'), ('A', 'phi3')])
