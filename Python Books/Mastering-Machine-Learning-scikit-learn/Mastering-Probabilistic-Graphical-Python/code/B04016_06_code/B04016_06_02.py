import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(100, 2))
raw_data
data = pd.DataFrame(raw_data, columns=['A', 'B'])
data

# Markov Model as stated in Fig. 6.5
markov_model = MarkovModel([('A', 'B')])
markov_model.fit(data, estimator=MaximumLikelihoodEstimator)
factors = coin_model.get_factors()
print(factors[0])
