import argparse

import chainer
import chainer.functions as cf
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


def clip(state):
    if isinstance(state, cp.ndarray):
        return cp.clip(state, 0, 1)
    if isinstance(state, np.ndarray):
        return np.clip(state, 0, 1)
    raise NotImplementedError()


def make_neighbor_state_pairs(states, x):
    if len(states) == 1:
        return [(x, states[0], None)]

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
    return neighbor_state_pairs


def main():
    # gpu/cpu
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    # model
    model = Model(ndim_hidden_units=[args.ndim_hidden] *
                  args.num_hidden_layers)
    if using_gpu:
        model.to_gpu()

    x = np.random.uniform(0, 1, size=(args.batchsize, 784)).astype(np.float32)
    y = np.random.uniform(0, 1, size=(args.batchsize, 10)).astype(np.float32)
    if using_gpu:
        x = cuda.to_gpu(x)
        y = cuda.to_gpu(y)

    states = model.generate_initial_states(args.batchsize)
    neighbor_state_pairs = make_neighbor_state_pairs(states, x)
    weight_pairs = model.forward_backward_weight_pairs()

    beta = 1000.0
    eps = 0.001

    # training loop
    for epoch in range(1000):
        ### First Phase ###
        for iteration in range(200):
            # compute vector field
            mu_array = []
            assert len(states) == len(weight_pairs)
            for (bottom_state, state, top_state), (Wf, Wb) in zip(
                    neighbor_state_pairs, weight_pairs):
                # output layer
                if Wb is None:
                    assert top_state is None
                    mu = rho(bottom_state).dot(Wf) - state
                    mu_array.append(mu)
                    continue

                # other layers
                mu = rho(bottom_state).dot(Wf) + rho(top_state).dot(Wb) - state
                mu_array.append(mu)

            # update states
            for state, mu in zip(states, mu_array):
                new_state = clip(state + eps * mu)
                state[...] = new_state

        # store states obtained in the first phase
        states_in_first_phase = []
        for state in states:
            states_in_first_phase.append(xp.copy(state))

        ### Second Phase ###
        for iteration in range(100):
            # compute gradient of the loss function with respect to x
            output_state = states[-1]
            variable = chainer.Variable(output_state)
            loss = cf.sum(0.5 * (y - variable)**2)
            variable.cleargrad()
            loss.backward()
            grad_objective = variable.grad

            # compute vector field
            mu_array = []
            assert len(states) == len(weight_pairs)
            for (bottom_state, state, top_state), (Wf, Wb) in zip(
                    neighbor_state_pairs, weight_pairs):
                # output layer
                if Wb is None:
                    assert top_state is None
                    mu = rho(bottom_state).dot(
                        Wf) - state - beta * grad_objective
                    mu_array.append(mu)
                    continue

                # other layers
                mu = rho(bottom_state).dot(Wf) + rho(top_state).dot(Wb) - state
                mu_array.append(mu)

            # update states
            for state, mu in zip(states, mu_array):
                new_state = clip(state + eps * mu)
                state[...] = new_state

        # store states obtained in the first phase
        states_in_second_phase = []
        for state in states:
            states_in_second_phase.append(xp.copy(state))

        # update weights
        mu_array = []
        neighbor_state_pairs_in_first_phase = make_neighbor_state_pairs(
            states_in_first_phase, x)
        assert len(states) == len(weight_pairs)
        for (bottom_state, state,
             top_state), (Wf, Wb), state_first, state_second in zip(
                 neighbor_state_pairs_in_first_phase, weight_pairs,
                 states_in_first_phase, states_in_second_phase):
            diff = state_second - state_first
            d = xp.expand_dims(diff, 1)

            r = xp.expand_dims(rho(bottom_state), 2)
            grad = xp.matmul(xp.expand_dims(r, 2), xp.expand_dims(d, 1))
            grad = xp.mean(grad, axis=(0, 1))
            Wf[...] += 0.01 * grad

            # output layer
            if Wb is None:
                assert top_state is None
                continue

            r = xp.expand_dims(rho(top_state), 2)
            grad = xp.matmul(xp.expand_dims(r, 2), xp.expand_dims(d, 1))
            grad = xp.mean(grad, axis=(0, 1))
            Wb[...] += 0.01 * grad

        # evaluate
        for iteration in range(200):
            # compute vector field
            mu_array = []
            assert len(states) == len(weight_pairs)
            for (bottom_state, state, top_state), (Wf, Wb) in zip(
                    neighbor_state_pairs, weight_pairs):
                # output layer
                if Wb is None:
                    assert top_state is None
                    mu = rho(bottom_state).dot(Wf) - state
                    mu_array.append(mu)
                    continue

                # other layers
                mu = rho(bottom_state).dot(Wf) + rho(top_state).dot(Wb) - state
                mu_array.append(mu)

            # update states
            for state, mu in zip(states, mu_array):
                new_state = clip(state + eps * mu)
                state[...] = new_state

        output_state = states[-1]
        loss = xp.sum((output_state - y)**2)
        print("output")
        print(output_state[-1])
        print("target")
        print(y[-1])
        print(loss)
        continue
        print("output distribution")
        print(output_state[-1] / xp.sum(output_state[-1]))
        print("target distribution")
        print(y[-1] / xp.sum(y[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--batchsize", "-bs", type=int, default=20)
    parser.add_argument("--ndim-hidden", "-ndim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", "-layers", type=int, default=1)
    args = parser.parse_args()
    main()
