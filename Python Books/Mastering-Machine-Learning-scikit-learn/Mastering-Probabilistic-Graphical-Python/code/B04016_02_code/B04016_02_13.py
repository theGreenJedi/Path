from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
# Creating the above bayesian network
model = BayesianModel()
model.add_nodes_from(['Rain', 'TrafficJam'])
model.add_edge('Rain', 'TrafficJam')
model.add_edge('Accident', 'TrafficJam')
cpd_rain = TabularCPD('Rain', 2, [[0.4], [0.6]])
cpd_accident = TabularCPD('Accident', 2, [[0.2], [0.8]])
cpd_traffic_jam = TabularCPD('TrafficJam', 2,
                             [[0.9, 0.6, 0.7, 0.1],
                              [0.1, 0.4, 0.3, 0.9]],
                             evidence=['Rain', 'Accident'],
                             evidence_card=[2, 2])
model.add_cpds(cpd_rain, cpd_accident, cpd_traffic_jam)
model.add_node('LongQueues')
model.add_edge('TrafficJam', 'LongQueues')
cpd_long_queues = TabularCPD('LongQueues', 2,
                             [[0.9, 0.2],
                              [0.1, 0.8]],
                             evidence=['TrafficJam'],
                             evidence_card=[2])
model.add_cpds(cpd_long_queues)
model.add_nodes_from(['GettingUpLate', 'LateForSchool'])
model.add_edges_from([('GettingUpLate', 'LateForSchool'),
                      ('TrafficJam', 'LateForSchool')])
cpd_getting_up_late = TabularCPD('GettingUpLate', 2,
                                 [[0.6], [0.4]])
cpd_late_for_school = TabularCPD('LateForSchool', 2,
                                 [[0.9, 0.45, 0.8, 0.1],
                                  [0.1, 0.55, 0.2, 0.9]],
                                 evidence=['GettingUpLate', 'TrafficJam'],
                                 evidence_card=[2, 2])
model.add_cpds(cpd_getting_up_late, cpd_late_for_school)
# Conversion from BayesianModel to MarkovModel is accomplished by
mm = model.to_markov_model()
mm.edges()
