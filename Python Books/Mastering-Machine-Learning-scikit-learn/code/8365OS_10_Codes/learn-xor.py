from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork, RecurrentNetwork, Network
from pybrain.supervised.trainers import BackpropTrainer

network = Network()
input_layer = LinearLayer(2)
hidden_layer = SigmoidLayer(5)
output_layer = LinearLayer(1)

network.addInputModule(input_layer)
network.addModule(hidden_layer)
network.addOutputModule(output_layer)

input_to_hidden = FullConnection(input_layer, hidden_layer)
hidden_to_output = FullConnection(hidden_layer, output_layer)
network.addConnection(input_to_hidden)
network.addConnection(hidden_to_output)
network.sortModules()

xor_dataset = SupervisedDataSet(2,1)
xor_dataset.addSample((0, 0), (0, ))
xor_dataset.addSample((0, 1), (1, ))
xor_dataset.addSample((1, 0), (1, ))
xor_dataset.addSample((1, 1), (0, ))

trainer = BackpropTrainer(module=network, dataset=xor_dataset, verbose=True,
                          momentum=0.00,
                          learningrate=0.10,
                          weightdecay=0.0,
                          lrdecay=1.0)

error = 1
epochsToTrain = 0
while error > 0.0001:
    epochsToTrain += 1
    error = trainer.train()

print ''
print 'Trained after', epochsToTrain, 'epochs'

# The network has been trained, now test it against our original data.
# Consider any number above 0.5 to be evaluated as 1, and below to be 0
print ''
print 'Final Results'
print '--------------'
results = network.activateOnDataset(xor_dataset)
for i in range(len(results)):
    print xor_dataset['input'][i], ' => ', (results[i] > 0.5), ' (',results[i],')'
