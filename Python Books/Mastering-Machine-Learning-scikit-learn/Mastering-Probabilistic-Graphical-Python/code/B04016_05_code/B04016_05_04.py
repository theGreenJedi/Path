import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
# Generating random data
raw_data = np.random.randint(low=0, high=2, size=(1000, 2))
data = pd.DataFrame(raw_data, columns=['X', 'Y'])
coin_model = BayesianModel()
coin_model.fit(data, estimator=MaximumLikelihoodEstimator)
coin_model.get_cpds()
coin_model.get_nodes()
coin_model.get_edges()