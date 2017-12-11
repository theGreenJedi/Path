from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
from pgmpy.inference import ClusterBeliefPropagation as CBP

# Create a bayesian model as we did in the previous chapters
model = BayesianModel([('rain', 'traffic_jam'),
                       ('accident', 'traffic_jam'),
                       ('traffic_jam', 'long_queues'),
                       ('traffic_jam', 'late_for_school'),
                       ('getting_up_late', 'late_for_school')])
cpd_rain = TabularCPD('rain', 2, [[0.4], [0.6]])
cpd_accident = TabularCPD('accident', 2, [[0.2], [0.8]])
cpd_traffic_jam = TabularCPD('traffic_jam', 2,
                             [[0.9, 0.6, 0.7, 0.1],
                              [0.1, 0.4, 0.3, 0.9]],
                             evidence=['rain', 'accident'],
                             evidence_card=[2, 2])
cpd_getting_up_late = TabularCPD('getting_up_late', 2,
                                 [[0.6], [0.4]])
cpd_late_for_school = TabularCPD('late_for_school', 2,
                                 [[0.9, 0.45, 0.8, 0.1],
                                  [0.1, 0.55, 0.2, 0.9]],
                                 evidence = ['getting_up_late',
                                             'traffic_jam'],
                                 evidence_card=[2, 2])
cpd_long_queues = TabularCPD('long_queues', 2,
                             [[0.9, 0.2],
                              [0.1, 0.8]],
                             evidence=['traffic_jam'],
                             evidence_card=[2])
model.add_cpds(cpd_rain, cpd_accident, cpd_traffic_jam,
               cpd_getting_up_late, cpd_late_for_school,
               cpd_long_queues)
cbp_inference = CBP(model)
cbp_inference.map_query(variables=['traffic_jam', 'late_for_school'])
cbp_inference.map_query(variables=['traffic_jam'],
                        evidence={'accident': 1, 'long_queues': 0})
