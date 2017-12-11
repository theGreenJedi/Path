model_inference.map_query(variables=['late_for_school'])
model_inference.map_query(variables=['late_for_school', 'accident'])
# Again we can pass the evidence to the query using the evidence
# argument in the form of {variable: state}.
model_inference.map_query(variables=['late_for_school'],
                          evidence={'accident': 1})
model_inference.map_query(variables=['late_for_school'],
                          evidence={'accident': 1, 'rain': 1})
# Also in the case of MAP queries we can specify the elimination
# order of the variables. But if the elimination order is not
# specified pgmpy automatically computes the best elimination
# order for the query.
model_inference.map_query(variables=['late_for_school'],
                          elimination_order=['accident', 'rain',
                                             'traffic_jam',
                                             'getting_up_late',
                                             'long_queues'])
model_inference.map_query(variables=['late_for_school'],
                          evidence={'accident': 1},
                          elimination_order=['rain',
                                             'traffic_jam',
                                             'getting_up_late',
                                             'long_queues'])
# Similarly MAP queries can be done for belief propagation as well.
belief_propagation.map_query(['late_for_school'],
                                      evidence={'accident': 1})