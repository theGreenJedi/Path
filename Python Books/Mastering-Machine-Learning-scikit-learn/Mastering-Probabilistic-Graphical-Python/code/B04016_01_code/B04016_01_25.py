from pgmpy.factors import RuleCPD
rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
                     ('A_1', 'B_0'): 0.2,
                     ('A_0', 'B_1', 'C_0'): 0.4,
                     ('A_1', 'B_1', 'C_0'): 0.6,
                     ('A_0', 'B_1', 'C_1'): 0.9,
                     ('A_1', 'B_1', 'C_1'): 0.1})
