from pgmpy.independencies import Independencies
# There are multiple ways to create an Independencies object, we
# could either initialize an empty object or initialize with some
# assertions.
independencies = Independencies() # Empty object
independencies.get_assertions()
independencies.add_assertions(assertion1, assertion2)
independencies.get_assertions()