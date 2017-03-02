import numpy as np

# A helper class, that allows getting the loss and
# gradients separately, without having to call the
# getter function twice.

# Not useful for FGSM, but works for any other gradient-based method.

class Eval(object):
    def __init__(self, f):
        self.l = None
        self.g = None
        self.f = f

    def loss(self, x):
        assert self.l is None
        l, g = self.f(x)
        self.l = l
        self.g = g
        return self.l

    def grads(self, x):
        assert self.l is not None
        g = np.copy(self.g)
        self.l = None
        self.g = None
        return g


