from pgmpy.factors import Factor
factor_a_b = Factor(variables=['A', 'B'], cardinality=[2, 2],
                    value=[90, 100, 1, 10])
factor_b_c = Factor(variables=['B', 'C'], cardinality=[2, 2],
                    value=[10, 80, 70, 30])
factor_c_d = Factor(variables=['C', 'D'], cardinality=[2, 2],
                    value=[10, 1, 100, 90])
factor_d_a = Factor(variables=['D', 'A'], cardinality=[2, 2],
                    value=[80, 60, 20, 10])
