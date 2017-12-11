# Firstly we need to import IndependenceAssertion
from pgmpy.independencies import IndependenceAssertion
# Each assertion is in the form of [X, Y, Z] meaning X is
# independent of Y given Z.
assertion1 = IndependenceAssertion('X', 'Y')
assertion1

