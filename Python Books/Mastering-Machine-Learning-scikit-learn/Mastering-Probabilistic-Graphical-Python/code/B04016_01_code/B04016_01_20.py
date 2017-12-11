model.add_node('long_queues')
model.add_edge('traffic_jam', 'long_queues')
cpd_long_queues = TabularCPD('long_queues', 2,
                             [[0.9, 0.2],
                              [0.1, 0.8]],
                             evidence=['traffic_jam'],
                             evidence_card=[2])
model.add_cpds(cpd_long_queues)
model.add_nodes_from(['getting_up_late', 'late_for_school'])
model.add_edges_from([('getting_up_late', 'late_for_school'),
                      ('traffic_jam', 'late_for_school')])
cpd_getting_up_late = TabularCPD('getting_up_late', 2,
                                 [[0.6], [0.4]])
cpd_late_for_school = TabularCPD('late_for_school', 2,
                                 [[0.9, 0.45, 0.8, 0.1],
                                  [0.1, 0.55, 0.2, 0.9]],
                                 evidence=['getting_up_late',
                                           'traffic_jam'],
                                 evidence_card=[2, 2])
model.add_cpds(cpd_getting_up_late, cpd_late_for_school)
model.get_cpds()
