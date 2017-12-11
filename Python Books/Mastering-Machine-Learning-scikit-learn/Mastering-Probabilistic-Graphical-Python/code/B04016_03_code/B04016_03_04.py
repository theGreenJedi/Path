# Firstly import JunctionTree class
from pgmpy.models import JunctionTree
junction_tree = JunctionTree()
# Each node in the junction tree is a cluster of random variables
# represented as a tuple
junction_tree.add_nodes_from([('A', 'B', 'C'), ('C', 'D')])
junction_tree.add_edge(('A', 'B', 'C'), ('C', 'D'))
junction_tree.add_edge(('A', 'B', 'C'), ('D', 'E', 'F'))