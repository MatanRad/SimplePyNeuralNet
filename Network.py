import numpy as np

class Network:
    def __init__(self,layerSizes, activator):
        # Initiate a vector so that sizes[l] = the number of neurons at layer l.
        self.sizes = [i for i in layerSizes]

        # Initiate the weights matrix so that w[l,k,j] = the weight of the connection from the j-th neuron in layer l-1 to the k-th neuron in layer l.
        self.weights = np.array([np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])])

        # Initiate the biases of the network such that b[i,j] = the bias of the j-th neuron in the l-th layer
        self.biases = np.array([np.random.randn(y,1) for y in layerSizes[1:]])

        # Initiate the activator function. needs to have an "Activate" function and "ActivateDeriv"
        self.activator = activator

        self.layersnum = len(layerSizes)

    def Test(self,testdata,testlabels):
        correct = 0.0
        for x,y in zip(testdata,testlabels):
            res = self.FeedForward(x)
            #print "shape res: " +str(res.shape)+"argmax: "+str(np.argmax(res)) + " y: "+str(y)
            if (np.argmax(res)==y): correct +=1
        return correct/len(testlabels)


    def FeedForward(self, a, track=False):
        zs = []
        if(track):
            al = [a]
            
        # b, w are the respective biases and weights for each iteration on each layer. Each iteration will calculate z[l+1] = the weighted input vector at the l+1-th layer (l is the layer of the current a vector) and then activate on it.
        for b,w in zip(self.biases, self.weights):
            zs.append(np.dot(w,a) + b)
            a = self.activator.Activate(zs[-1])
            if (track): al.append(a)
        if (track): return (a,al,zs)
        return a