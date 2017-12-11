import numpy as np
import pandas as pd
from pgmpy.models import MarkovModel
from pgmpy.estimators import PseudoMomentMatchingEstimator
# Generating some random data
raw_data = np.random.randint(low=0, high=2, size=(100, 4))
raw_data
data = pd.DataFrame(raw_data, columns=['A', 'B', 'C', 'D'])
data

# Diamond shaped Markov Model as stated in Fig. 6.1
markov_model = MarkovModel([('A', 'B'), ('B', 'C'),
                            ('C', 'D'), ('D', 'A')])
markov_model.fit(data, estimator=PseudoMomentMatchingEstimator)
factors = coin_model.get_factors()
factors
