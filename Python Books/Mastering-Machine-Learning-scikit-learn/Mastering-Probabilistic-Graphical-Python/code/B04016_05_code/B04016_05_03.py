import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(1000, 6))
print(raw_data)
data = pd.DataFrame(raw_data, columns=['A', 'R', 'J', 'G', 'L', 'Q'])

# Creating the network structures
student_model = BayesianModel([('A', 'J'), ('R', 'J'),
                               ('J', 'Q'), ('J', 'L'),
                               ('G', 'L')])
student_model.fit(data, estimator=BayesianEstimator)
student_model.get_cpds()
print(student_model.get_cpds('D'))
