from pgmpy.models import MarkovModel
mm = MarkovModel()
mm.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
mm.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
                   ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
                   ('x4', 'x7'), ('x5', 'x7')])
mm.get_local_independencies()
