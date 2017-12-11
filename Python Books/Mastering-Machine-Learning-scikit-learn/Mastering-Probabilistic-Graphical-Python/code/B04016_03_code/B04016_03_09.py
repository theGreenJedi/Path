from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD
from pgmpy.inference import VariableElimination
# Constructing the model
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
                             evidence=['rain',
                                       'accident'],
                             evidence_card=[2, 2])
cpd_getting_up_late = TabularCPD('getting_up_late', 2, [[0.6], [0.4]])
cpd_late_for_school = TabularCPD('late_for_school', 2,
                                 [[0.9, 0.45, 0.8, 0.1],
                                  [0.1, 0.55, 0.2, 0.9]],
                                 evidence=['getting_up_late', 'traffic_jam'],
                                 evidence_card=[2, 2])
cpd_long_queues = TabularCPD('long_queues', 2,
                             [[0.9, 0.2],
                              [0.1, 0.8]],
                             evidence=['traffic_jam'],
                             evidence_card=[2])
model.add_cpds(cpd_rain, cpd_accident,
               cpd_traffic_jam, cpd_getting_up_late,
               cpd_late_for_school, cpd_long_queues)
# Calculating max marginals
model_inference = VariableElimination(model)
model_inference.max_marginal(variables=['late_for_school'])
model_inference.max_marginal(variables=['late_for_school', 'traffic_jam'])
# For any evidence in the network we can simply pass the evidence
# argument which is a dict of the form of {variable: state}
model_inference.max_marginal(variables=['late_for_school'],
                             evidence={'traffic_jam': 1})
model_inference.max_marginal(variables=['late_for_school'],
                             evidence={'traffic_jam': 1,
                                       'getting_up_late': 0})
model_inference.max_marginal(variables=['late_for_school','long_queues'],
                             evidence={'traffic_jam': 1,
                                       'getting_up_late': 0}
# Again as in the case of VariableEliminaion we can also pass the
# elimination order of variables for MAP queries. If not specified
# pgmpy automatically computes the best elimination order for the
# query.
model_inference.max_marginal(variables=['late_for_school'],
                             elimination_order=['traffic_jam',
                                                'getting_up_late', 'rain',
                                                'long_queues', 'accident'])
model_inference.max_marginal(variables=['late_for_school'],
                             evidence={'accident': 1},
                             elimination_order=['traffic_jam',
                                                'getting_up_late',
                                                'rain', 'long_queues'])