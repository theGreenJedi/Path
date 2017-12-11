from pgmpy.factors import TabularCPD
cpd_rain = TabularCPD('rain', 2, [[0.4], [0.6]])
cpd_accident = TabularCPD('accident', 2, [[0.2], [0.8]])
cpd_traffic_jam = TabularCPD('traffic_jam', 2,
                             [[0.9, 0.6, 0.7, 0.1],
                              [0.1, 0.4, 0.3, 0.9]],
                             evidence=['rain', 'accident'],
                             evidence_card=[2, 2])
