from pgmpy.factors import Factor
from pgmpy.factors import FactorSet
from pgmpy.models import MarkovModel
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
import functools

def compute_message(cluster_1, cluster_2,
                    inference_data_structure=VariableElimination):
    """
    Computes the message from cluster_1 to cluster_2.
    The messages are computed by projecting a factor set to
    produce a set of marginals over a given set of scopes. The
    factor set is nothing but the factors present in the models.
    The algorithm for computing messages between any two clusters
    is:
    * Build an inference data structure with all the factors
    represented in the cluster.
    * Perform inference on the cluster using the inference data
    structure to compute the marginals of the variables present
    in the sepset between these two clusters.
    * The output message is the factor set of all the computed
    marginals.
    Parameters
    ----------
    cluster_1: MarkovModel, BayesianModel, or any pgmpy supported
    graphical model
    The cluster producing the message
    cluster_2: MarkovModel, BayesianModel, or any pgmpy supported
    graphical model
    The cluster receiving the message
    inference_data_structure: Inference class such as
    VariableElimination or BeliefPropagation
    The inference data structure used to produce factor
    set of marginals
    """
    # Sepset variables between the two clusters
    sepset_var = set(cluster_1.nodes()).intersection(
    cluster_2.nodes())
    # Initialize the inference data structure
    inference = inference_data_structure(cluster_1)
    # Perform inference
    query = inference.query(list(sepset_var))

    # The factor set of all the computed messages is the output
    # message query would be a dictionary with key as the variable
    # and value as the corresponding marginal thus the values
    # would represent the factor set
    return FactorSet(*query.values())


def compute_belief(cluster, *input_factored_messages):
    """
    Computes the belief a particular cluster given the cluster
    and input messages
    \delta_{j \rightarrow i} where j are all the neighbors of
    cluster i. The cluster belief is computed as:
    .. math::
    \beta_i = \psi_i \prod_{j \in Nb_i} \delta_{j \rightarrow i}
    where \psi_i is the cluster potential. As the cluster belief
    represents the probability and it should be normalized to sum
    up to 1.
    Parameters
    ----------
    cluster: MarkovModel, BayesianModel, or any pgmpy supported
    graphical model
    The cluster whose cluster potential is going to be
    computed.
    *input_factored_messages: FactorSet or a group of FactorSets
    All the input messages to the clusters. They should be
    factor sets
    Returns
    -------
    cluster_belief: Factor
    The cluster belief of the corresponding cluster
    """
    messages_prod = functools.reduce(lambda x, y: x * y,
    input_factored_messages)
    # As messages_prod would be a factor set, so its corresponding
    # factor would be product of all the factors present in the
    # factorset
    messages_prod_factor = functools.reduce(lambda x, y: x * y,
    messages_prod.factors)
    # Computing cluster potential psi
    psi = functools.reduce(lambda x, y: x * y,
    cluster.get_factors())
    # As psi represents the probability it should be normalized
    psi.normalize()
    # Computing the cluster belief according the formula stated
    # above
    cluster_belief = psi * messages_prod_factor
    # As cluster belief represents a probability distribution in
    # this case, thus it should be normalized
    cluster_belief.normalize()
    return cluster_belief


phi_a_b = Factor(['a', 'b'], [2, 2], [10, 0.1, 0.1, 10])
phi_a_c = Factor(['a', 'c'], [2, 2], [5, 0.2, 0.2, 5])
phi_c_d = Factor(['c', 'd'], [2, 2], [0.5, 1, 20, 2.5])
phi_d_b = Factor(['d', 'b'], [2, 2], [5, 0.2, 0.2, 5])

# Cluster 1 is a MarkovModel A--B
cluster_1 = MarkovModel([('a', 'b')])

# Adding factors
cluster_1.add_factors(phi_a_b)

# Cluster 2 is a MarkovModel A--C--D--B
cluster_2 = MarkovModel([('a', 'c'), ('c', 'd'), ('d', 'b')])

# Adding factors
cluster_2.add_factors(phi_a_c, phi_c_d, phi_d_b)

# Message passed from cluster 1 -> 2 should the M-Projection of psi1
# as the sepset of cluster 1 and 2 is A, B thus there is no need to
# marginalize psi1
delta_1_2 = compute_message(cluster_1, cluster_2)

# If we want to use any other inference data structure we can pass
# them as an input argument such as: delta_1_2 =
# compute_message(cluster_1, cluster_2, BeliefPropagation)
beta_2 = compute_belief(cluster_2, delta_1_2)
print(beta_2.marginalize(['a', 'b'], inplace=False))

# Lets compute the belief of cluster1, first we need to compute the
# output message from cluster 2 to cluster 1
delta_2_1 = compute_message(cluster_2, cluster_1)

# Lets see the distribution of both of these variables in the
# computed message
for phi in delta_2_1.factors:
    print(phi)

# The belief of cluster1 would be
beta_1 = compute_belief(cluster_1, delta_2_1)
print(beta_1)
