import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator
# Generating random data
raw_data = np.random.randint(low=0, high=2, size=(1000, 2))
data = pd.DataFrame(raw_data, columns=['X', 'Y'])
model = MarkovModel()
model.fit(data, estimator=MaximumLikelihoodEstimator)
model.get_factors()
model.nodes()
model.edges()
