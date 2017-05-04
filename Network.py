import numpy as np

class Network:
	def __init__(self,layerSizes, activator):
		# Initiate a vector so that sizes[l] = the number of neurons at layer l.
		self.sizes = [i for i in layerSizes]

		# Initiate the weights matrix so that w[l,k,j] = the weight of the connection from the j-th neuron in layer l-1 to the k-th neuron in layer l.
		self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

		# Initiate the biases of the network such that b[i,j] = the bias of the j-th neuron in the l-th layer
		self.biases = [np.random.randn(y,1) for y in layerSizes]

		# Initiate the activator function. needs to have an "Activate" function and "ActivateDeriv"
		self.activator = activator

	def FeedForward(self, a):
		# b, w are the respective biases and weights for each iteration on each layer. Each iteration will calculate z[l+1] = the weighted input vector at the l+1-th layer (l is the layer of the current a vector) and then activate on it.
		for b,w in zip(self.biases, self.weights):
			a = self.activator.Activate(np.dot(w,a)+b)
		return a

