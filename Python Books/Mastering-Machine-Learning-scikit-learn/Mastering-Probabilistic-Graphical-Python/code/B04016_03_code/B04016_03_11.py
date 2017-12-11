# First let's import modules that we will be needing
import numpy as np
from pgmpy.models import BayesianModel
# Now let's create some random data over which we will train and
# test the model. Here we are creating 1000 data points with each
# value either 0 or 1.
data = np.random.randint(low=0, high=2, size=(1000, 4))
data
# Now in general machine learning problems it doesn't matter which
# column of the array represents which variable (until we use same
# order for both training and prediction) because all the values
# are on symmetrical axis but in graphical models each variable is
# different (in the way it is connected to other variables etc) so
# we will need to specify which columns of data are for which
# variable. For that we will use pandas.
import pandas as pd
data = pd.DataFrame(data, columns=['cost', 'quality',
                                   'location', 'no_of_people'])
data
train = data[:750]
# We will try to predict the no_of_people from our model. So for
# test data we will delete that column and then later on predict
# those values.
test = data[750:].drop('no_of_people', axis=1)
test
# Now we will need to create the base network structure for the
# model.
restaurant_model = BayesianModel([('location', 'cost'),
                                  ('quality', 'cost'),
                                  ('location', 'no_of_people'),
                                  ('cost', 'no_of_people')])
restaurant_model.fit(train)
# Fit computes the cpd of all the variables from the training data
# that we provided.
restaurant_model.get_cpds()
# Now for predicting the values of no_of_people using this model
# we can simply call the predict method on our test data.
restaurant_model.predict(test).values.ravel()
