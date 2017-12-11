# mean of the emission probability distribution for various states
# were:
# [0.0, 0.0],
# [0.0, 10.0],
# [10.0, 0.0]
# So if an observations are sampled from some other gaussian
# distribution with mean centered at different location such as:
# [5, 5]
# [-5, 0]
# the probability of these observations coming from this model
# should be very low.
# generating observations
observations = np.row_stack(
    (np.random.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], 10),
     np.random.multivariate_normal([-5, 0], [[0.5, 0], [0, 0.5]], 10)))

# model.score returns the log-probability of P(observations |
# model)
score_1 = model_gaussian.score(observations)
print(score_1)

# Lets try to check whether observations sampled from the
# multivariate normal distributions that were used in our HMM
# model provides greater value of score or not
observations = np.row_stack((
    np.random.multivariate_normal([10, 0], [[0.5, 0], [0, 0.5]], 10),
    np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 2),
    np.random.multivariate_normal([0, 10], [[0.5, 0], [0, 0.5]], 4)))

score_2 = model_gaussian.score(observations)
print(score_2)