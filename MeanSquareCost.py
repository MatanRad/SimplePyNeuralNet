import numpy as np

class MeanSquareCost:
    def Calculate(self, a, targetvec):
        d = (targetvec-a)
        return 0.5*(d*d)

    def CalculateDeriv(self, a, targetvec):
        return a-targetvec
