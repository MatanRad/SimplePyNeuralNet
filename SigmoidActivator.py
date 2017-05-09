import numpy as np

class SigmoidActivator:
	def Activate(self, z):
		#print "z: " + str(z) + " a " + str(1/(1+np.exp(-z)))
		return 1/(1+np.exp(-z))

	def ActivateDeriv(self, z):
		return self.ActivatedDeriv(Activate(z))

	def ActivatedDeriv(self, a):
		return a*(1-a)