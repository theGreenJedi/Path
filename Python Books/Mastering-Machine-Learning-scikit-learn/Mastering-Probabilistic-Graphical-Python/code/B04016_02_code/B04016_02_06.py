# First import MarkovModel class from pgmpy.models
from pgmpy.models import MarkovModel
model = MarkovModel([('A', 'B'), ('B', 'C')])
model.add_node('D')
model.add_edges_from([('C', 'D'), ('D', 'A')])
