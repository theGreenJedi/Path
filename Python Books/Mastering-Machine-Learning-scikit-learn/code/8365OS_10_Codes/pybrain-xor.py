# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.datasets import SupervisedDataSet
# from pybrain.structure import SigmoidLayer, LinearLayer, FullConnection, FeedForwardNetwork
#
# net = FeedForwardNetwork()
# input_layer = LinearLayer(2)
# hidden_layer = SigmoidLayer(3)
# output_layer = LinearLayer(1)
#
# net.addInputModule(input_layer)
# net.addModule(hidden_layer)
# net.addOutputModule(output_layer)
#
# input_to_hidden = FullConnection(input_layer, hidden_layer)
# hidden_to_output = FullConnection(hidden_layer, output_layer)
# net.addConnection(input_to_hidden)
# net.addConnection(hidden_to_output)
# net.sortModules()
#
# xor_dataset = SupervisedDataSet(2, 1)
# xor_dataset.addSample((0, 0), (0, ))
# xor_dataset.addSample((0, 1), (1, ))
# xor_dataset.addSample((1, 0), (1, ))
# xor_dataset.addSample((1, 1), (0, ))
#
# trainer = BackpropTrainer(module=net, dataset=xor_dataset, verbose=True,
#                           momentum=0.00,
#                           learningrate=0.10,
#                           weightdecay=0.0,
#                           lrdecay=1.0)
#
# # Train until the error is below a certain threshold.
# # This training method uses the entire dataset instead of just a
# # portion as the trainUntilConvergence method does
# error = 1
# epochsToTrain = 0
# while (error > 0.0001:
#     epochsToTrain += 1
#     error = trainer.train()
#
# print ''
# print 'Trained after', epochsToTrain, 'epochs'
#
# print ''
# print 'Final Results'
# print '--------------'
# results = net.activateOnDataset(xor_dataset)
# for i in range(len(results)):
#     print xor_dataset['input'][i], ' => ', (results[i] > 0.5), ' (',results[i],')'
