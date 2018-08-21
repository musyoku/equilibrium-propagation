import chainer
import chainer.functions as cf
import chainer.links as L
from chainer.initializers import GlorotNormal


class Model(chainer.Chain):
    def __init__(self, ndim_hidden=512, num_hidden_layers=1):
        super().__init__()
        self.ndim_hidden = ndim_hidden
        self.num_hidden_layers = num_hidden_layers

        forward_weights = chainer.ChainList()
        backward_weights = chainer.ChainList()

        with forward_weights.init_scope():
            forward_weights.append(
                L.Linear(784, ndim_hidden, initialW=GlorotNormal(1.0)))
            for _ in range(num_hidden_layers):
                forward_weights.append(
                    L.Linear(
                        ndim_hidden, ndim_hidden, initialW=GlorotNormal(1.0)))
            forward_weights.append(
                L.Linear(ndim_hidden, 10, initialW=GlorotNormal(1.0)))

        with backward_weights.init_scope():
            backward_weights.append(
                L.Linear(10, ndim_hidden, initialW=GlorotNormal(1.0)))
            for _ in range(num_hidden_layers):
                backward_weights.append(
                    L.Linear(
                        ndim_hidden, ndim_hidden, initialW=GlorotNormal(1.0)))

        with self.init_scope():
            self.forward_weights = forward_weights
            self.backward_weights = backward_weights

    def generate_initial_states(self, batchsize):
        xp = self.xp

        # input
        states = [xp.random.uniform(0, 1, size=(batchsize, 784))]

        # hidden
        for _ in range(self.num_hidden_layers):
            states.append(
                xp.random.uniform(0, 1, size=(batchsize, self.ndim_hidden)))
        # output
        states.append(xp.random.uniform(0, 1, size=(batchsize, 10)))

        return states