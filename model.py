import chainer
import chainer.functions as cf
import chainer.links as L
from chainer.initializers import GlorotNormal


class Model(chainer.Chain):
    def __init__(self, ndim_hidden_units=[512]):
        super().__init__()
        assert len(ndim_hidden_units) > 0
        self.ndim_hidden_units = ndim_hidden_units

        forward_links = chainer.ChainList()
        backward_links = chainer.ChainList()

        with forward_links.init_scope():
            ndim_units = [784] + ndim_hidden_units + [10]
            for in_units, out_units in zip(ndim_units[:-1], ndim_units[1:]):
                forward_links.append(
                    L.Linear(
                        in_units,
                        out_units,
                        nobias=True,
                        initialW=GlorotNormal(1.0)))

        with backward_links.init_scope():
            ndim_units = ndim_hidden_units + [10]
            for out_units, in_units in zip(ndim_units[:-1], ndim_units[1:]):
                backward_links.append(
                    L.Linear(
                        in_units,
                        out_units,
                        nobias=True,
                        initialW=GlorotNormal(1.0)))

        with self.init_scope():
            self.forward_links = forward_links
            self.backward_links = backward_links

    def forward_backward_weight_pairs(self):
        # e.g.
        # network: x -> s1 -> s2 -> s3 -> y
        # W_ij: j to i
        # forward_weights = [W_10, W_21, W_32]
        # backward_weights = [W_12, W_23]

        forward_weights = []
        backward_weights = []

        for link in self.forward_links.children():
            weight = link.W.data
            forward_weights.append(weight)

        for link in self.backward_links.children():
            weight = link.W.data
            backward_weights.append(weight)
        backward_weights.append(None)

        forward_backward_weight_pairs = []
        for Wf, Wb in zip(forward_weights, backward_weights):
            forward_backward_weight_pairs.append((Wf, Wb))

        # [(W_10, W_12), (W_21, W_23), (W_32, None)]
        return forward_backward_weight_pairs

    def generate_initial_states(self, batchsize):
        xp = self.xp

        states = []

        ndim_units = self.ndim_hidden_units + [10]
        for ndim in ndim_units:
            states.append(xp.random.uniform(0, 1, size=(batchsize, ndim)))

        return states