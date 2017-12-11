from pgmpy.factors import TreeCPD, Factor
tree_cpd = TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), '0'),
                    ('B', 'C', '1'),
                    ('C', Factor(['A'], [2], [0.1, 0.9]), '0'),
                    ('C', 'D', '1'),
                    ('D', Factor(['A'], [2], [0.9, 0.1]), '0'),
                    ('D', Factor(['A'], [2], [0.4, 0.6]), '1')])
