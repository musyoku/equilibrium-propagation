import argparse

import cupy as cp
import numpy as np
from chainer.backends import cuda

from model import Model


def rho(state):
    if isinstance(state, cp.ndarray):
        return cp.clip(state, 0, 1)
    if isinstance(state, np.ndarray):
        return np.clip(state, 0, 1)
    raise NotImplementedError()


def main():
    # gpu/cpu
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    # model
    model = Model(ndim_hidden_units=[123, 345, 567])
    if using_gpu:
        model.to_gpu()

    x = np.random.normal(size=(args.batchsize, 784))
    y = np.random.normal(size=(args.batchsize, 10))

    # training loop
    states = model.generate_initial_states(args.batchsize)
    if len(states) == 1:
        neighbor_state_pairs = [(x, states[0], None)]
    else:
        neighbor_state_pairs = []
        for layer_index, state in enumerate(states):
            if layer_index == 0:
                # input layer
                top_state = states[layer_index + 1]
                bottom_state = x
                neighbor_state_pairs.append((bottom_state, state, top_state))
                continue
            if layer_index == len(states) - 1:
                # output layer
                bottom_state = states[layer_index - 1]
                neighbor_state_pairs.append((bottom_state, state, None))
                continue
            # hidden layer
            top_state = states[layer_index + 1]
            bottom_state = states[layer_index - 1]
            neighbor_state_pairs.append((bottom_state, state, top_state))

    # compute vector field
    mu_array = []
    weight_pairs = model.forward_backward_weight_pairs()

    # print("states")
    # for state in states:
    #     print(state.shape)

    # print("weight_pairs")
    # for (Wf, Wb) in weight_pairs:
    #     if Wb is None:
    #         print(Wf.shape)
    #     else:
    #         print(Wf.shape, Wb.shape)

    assert len(states) == len(weight_pairs)
    for (bottom_state, state, top_state), (Wf, Wb) in zip(
            neighbor_state_pairs, weight_pairs):
        if Wb is None:
            assert top_state is None
            mu = rho(bottom_state).dot(Wf.T) - state
            mu_array.append(mu)
            continue
        mu = rho(bottom_state).dot(Wf.T) + rho(top_state).dot(Wb.T) - state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--batchsize", "-bs", type=int, default=20)
    parser.add_argument("--ndim-hidden", "-ndim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", "-layers", type=int, default=1)
    args = parser.parse_args()
    main()
