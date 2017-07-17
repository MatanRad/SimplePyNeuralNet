import numpy as np

class Trainer:
    def __init__(self, network):
        self.network = network

    def Backprop(self, x,targetvec, cost):
        out,activations,zs = self.network.FeedForward(x, True)
        dweights = [np.zeros(y.shape) for y in self.network.weights]
        dbiases = [np.zeros(y.shape) for y in self.network.biases]


        dbiases[-1] = cost.CalculateDeriv(activations[-1], targetvec) * self.network.activator.ActivatedDeriv(zs[-1])
        dweights[-1] = np.dot(dbiases[-1],activations[-2].transpose())
        #dweights[-1] = dbiases[-1]*activations[-2]
        #print "db shape "+str(dbiases[-1].shape) + " a " + str(len(activations))
        l = len(dbiases)
        for i in reversed(range(1,len(dbiases)-1)):
            dbiases[i] = np.dot(self.network.weights[i-1].transpose(), dbiases[i+1]) * self.network.activator.ActivatedDeriv(zs[i])
            dweights[i] = np.dot(dbiases[i],activations[i-1].transpose())
            

        return dbiases,dweights

    def SGD(self, dataset, datalabels, cost, alpha, epochs, batchsize, testpair = None, stoppercent=None):
        trainingset = [[x,y] for x,y in zip(dataset, datalabels)]

        for epoch in xrange(epochs):
            np.random.shuffle(trainingset)
            batches = [trainingset[k:k+batchsize] for k in range(0, len(trainingset),batchsize)]

            onehot = np.zeros((self.network.biases[-1].shape[0],1))

            for batch in batches:
                nabla_b = [np.zeros(b.shape) for b in self.network.biases]
                nabla_w = [np.zeros(w.shape) for w in self.network.weights]
                for x,y in batch:
                    currbatchsize = len(batch)

                    onehot[y] = 1
                    deltab, deltaw = self.Backprop(x,onehot,cost)
                    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, deltab)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, deltaw)]

                    onehot[y] = 0
                self.network.weights = [w-(alpha/len(batch))*nw for w, nw in zip(self.network.weights, nabla_w)]
                self.network.biases = [b-(alpha/len(batch))*nb for b, nb in zip(self.network.biases, nabla_b)]

            if testpair != None:
                percent = self.network.Test(testpair[0],testpair[1])
                print "Epoch "+str(epoch)+": "+str(percent)
                if (stoppercent!=None): 
                    if (percent>stoppercent): break
                #self.network.weights -= (alpha/currbatchsize) * deltaw
                #self.network.biases -= (alpha/currbatchsize) * deltab
