class Trainer:
	def __init__(self, network):
		self.network = network

	def Backprop(self, x,targetvec, cost):
		out,activations = self.network.FeedForward(x, True)
		dweights = np.empty(self.network.weights.shape)
		dbiases = np.empty(self.network.biases.shape)



		dbiases[-1] = cost.CalculateDeriv(activations[self.network.layersnum-1], targetvec) * self.network.activator.ActivatedDeriv(activations[self.network.layersnum-1])
		dweights[-1] = np.dot(dbiases[self.network.layersnum-1],activations[self.network.layersnum-1].transpose())
		
		for i in reversed(range(self.network.layersnum-1)):
			dbiases[i] = np.dot(self.netowrk.weights[i].transpose(), dbiases[i+1]) * self.network.activator.ActivatedDeriv(activations[i+1])
			if (i!=0) dweights[i-1] = np.dot(dbiases[i],activations[i].transpose())

		return dbiases,dweights

	