import numpy as np

class SigmoidActivator:
	def Activate(self, z):
		return 1/(1+np.exp(-z))

	def ActivateDeriv(self, z):
		return self.ActivatedDeriv(Activate(z))

	def ActivatedDeriv(self, a):
		return a*(1-a)