cost = TabularCPD(variable='Cost',
                  variable_card=2,
                  values=[[0.8, 0.6, 0.1, 0.6, 0.6, 0.05],
                          [0.2, 0.4, 0.9, 0.4, 0.4, 0.95]],
                  evidence=['Q', 'L'],
                  evidence_card=[3, 2])
