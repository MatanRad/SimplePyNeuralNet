import Network, Trainer, DatasetLoader, MeanSquareCost, SigmoidActivator
import os
import numpy as np
import cPickle as pickle

ptrain = "trainset.pickle"
ptest = "testset.pickle"
pnetwork = "network.pickle"

dataset_folder = "C:\\Users\\matan\\Desktop\\heavyprojects\\notMNIST_large"
testset_folder = "C:\\Users\\matan\\Desktop\\heavyprojects\\notMNIST_small"

initialize = True
if os.path.exists(pnetwork):
    print "Should I load the existing network? y/n\n"
    if raw_input()=='y':
        with open(pnetwork, "rb") as f:
            print "Loading Network"
            network = pickle.load(f)
            initialize = False

if initialize:
    activator = SigmoidActivator.SigmoidActivator()
    network = Network.Network((784,15,10),activator)
    
    with open(pnetwork, "wb") as f:
        pickle.dump(network,f)




test = []
train = []
if os.path.exists(ptrain):
    print "Reading training set"
    with open(ptrain, "rb") as f:
        train = pickle.load(f)
else:
    dataset, datalabels = DatasetLoader.LoadRandomizedDataset(dataset_folder,2000,True)
    train = (dataset,datalabels)

    print "Writing training set"
    with open(ptrain, "wb") as f:
        pickle.dump(train,f)

if os.path.exists(ptest):
    print "Reading test set"
    with open(ptest, "rb") as f:
        test = pickle.load(f)
else:
    testset, testlabels = DatasetLoader.LoadRandomizedDataset(testset_folder,2000,True)
    test = (testset, testlabels)

    print "Writing test set"
    with open(ptest, "wb") as f:
        pickle.dump(test,f)

print "Training and test sets ready"

trainer = Trainer.Trainer(network)
#print network.Test(*test)
cost = MeanSquareCost.MeanSquareCost()

print "Training Set Shape: "+str(train[0][0].shape)
print "Training the network"
trainer.SGD(train[0],train[1],cost,0.1,100,50, test, 0.8)
print "Saving the network"
with open(pnetwork, "wb") as f:
        pickle.dump(network,f)

print "Success rate: "+str(network.Test(*test)*100)+"%"