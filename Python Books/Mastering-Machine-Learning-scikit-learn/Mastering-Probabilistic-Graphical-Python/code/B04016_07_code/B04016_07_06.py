from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import numpy as np

# Here n_components correspond to number of states in the hidden
# variables.
model_gaussian = GaussianHMM(n_components=3, covariance_type='full')

# Transition probability as specified above
transition_matrix = np.array([[0.2, 0.6, 0.2],
                              [0.4, 0.3, 0.3],
                              [0.05, 0.05, 0.9]])

# Setting the transition probability
model_gaussian.transmat_ = transition_matrix

# Initial state probability
initial_state_prob = np.array([0.1, 0.4, 0.5])

# Setting initial state probability
model_gaussian.startprob_ = initial_state_prob

# As we want to have a 2-D gaussian distribution the mean has to
# be in the shape of (n_components, 2)
mean = np.array([[0.0, 0.0],
                 [0.0, 10.0],
                 [10.0, 0.0]])

# Setting the mean
model_gaussian.means_ = mean

# As emission probability is a 2-D gaussian distribution, thus
# covariance matrix for each state would be a 2-D matrix, thus
# overall the covariance matrix for all the states would be in the
# form of (n_components, 2, 2)
covariance = 0.5 * np.tile(np.identity(2), (3, 1, 1))
model_gaussian.covars_ = covariance

# model.sample returns both observations as well as hidden states
# the first return argument being the observation and the second
# being the hidden states
Z, X = model_gaussian.sample(100)

# Plotting the observations
plt.plot(Z[:, 0], Z[:, 1], "-o", label="observations",
         ms=6, mfc="orange", alpha=0.7)

# Indicate the state numbers
for i, m in enumerate(mean):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=17, horizontalalignment='center',
             bbox=dict(alpha=.7, facecolor='w'))
plt.legend(loc='best')
plt.show()
