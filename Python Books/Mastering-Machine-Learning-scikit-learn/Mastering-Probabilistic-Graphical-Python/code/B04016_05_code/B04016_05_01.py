import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(100, 2))
print(raw_data)
data = pd.DataFrame(raw_data, columns=['X', 'Y'])
print(data)

# Two coin tossing model assuming that they are dependent.
coin_model = BayesianModel([('X', 'Y')])
coin_model.fit(data, estimator=MaximumLikelihoodEstimator)
cpd_x = coin_model.get_cpds('X')
print(cpd_x)
