import numpy as np

class Trainer:
	def __init__(self, network):
		self.network = network

	def Backprop(self, x,targetvec, cost):
		out,activations = self.network.FeedForward(x, True)
		dweights = np.array([np.empty(y.shape) for y in self.network.weights])
		dbiases = np.array([np.empty(y.shape) for y in self.network.biases])

		dbiases[-1] = cost.CalculateDeriv(activations[-1], targetvec) * self.network.activator.ActivatedDeriv(activations[-1])
		dweights[-1] = np.dot(dbiases[-1],activations[-2].transpose())
		#print "db shape "+str(dbiases[-1].shape) + " a " + str(len(activations))

		for i in reversed(range(len(dbiases)-1)):
			dbiases[i] = np.dot(self.network.weights[i+1].transpose(), dbiases[i+1]) * self.network.activator.ActivatedDeriv(activations[i+1])
			if (i!=0): dweights[i-1] = np.dot(dbiases[i],activations[i-1].transpose())
			

		return dbiases,dweights

	def SGD(self, dataset, datalabels, cost, alpha, epochs, batchsize):
		trainingset = [[x,y] for x,y in zip(dataset, datalabels)]

		for epoch in xrange(epochs):
			np.random.shuffle(trainingset)
			batches = [trainingset[k:k+batchsize] for k in range(0, len(trainingset),batchsize)]

			onehot = np.zeros((self.network.biases[-1].shape[0],1))

			for batch in batches:

				for x,y in batch:
					currbatchsize = len(batch)

					onehot[y] = 1
					deltab, deltaw = self.Backprop(x,onehot,cost)
					onehot[y] = 0
					self.network.weights -= (alpha/currbatchsize) * deltaw
					self.network.biases -= (alpha/currbatchsize) * deltab
