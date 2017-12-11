from pgmpy.factors import JointProbabilityDistribution as Joint
distribution = Joint(['coin1', 'coin2'],
                     [2, 2],
                     [0.25, 0.25, 0.25, 0.25])
