raw_data = np.random.randint(low=0, high=2, size=(1000, 6))
data = pd.DataFrame(raw_data, columns=['A', 'R', 'J', 'G', 'L', 'Q'])
student_model = BayesianModel([('A', 'J'), ('R', 'J'),
                               ('J', 'Q'), ('J', 'L'),
                               ('G', 'L')])
student_model.fit(data, estimator=MaximumLikelihoodEstimator)
student_model.get_cpds()
