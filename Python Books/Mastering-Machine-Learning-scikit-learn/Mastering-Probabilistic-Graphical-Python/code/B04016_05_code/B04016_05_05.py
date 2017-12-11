import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
# Generating random data for two coin tossing examples
raw_data = np.random.randint(low=0, high=2, size=(1000, 2))
data = pd.DataFrame(raw_data, columns=['X', 'Y'])
print(data)
coin_model = BayesianModel()
coin_model.fit(data, estimator=BayesianEstimator)
coin_model.get_cpds()
coin_model.nodes()
coin_model.edges()