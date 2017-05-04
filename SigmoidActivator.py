import numpy as np

class SigmoidActivator:
	def Activate(self, z):
		return 1/(1+np.exp(-z))

	def ActivateDeriv(self, ):
		a = Activate(z)
		return a*(1-a)