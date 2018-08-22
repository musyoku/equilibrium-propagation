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

    x = np.random.uniform(0, 1, size=(args.batchsize, 784)).astype(np.float32)
    y = np.random.uniform(0, 1, size=(args.batchsize, 10)).astype(np.float32)
    if using_gpu:
        x = cuda.to_gpu(x)
        y = cuda.to_gpu(y)

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

    weight_pairs = model.forward_backward_weight_pairs()

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
            eps = 0.001
            for state, mu in zip(states, mu_array):
                new_state = clip(state + eps * mu)
                state[...] = new_state

        # store states obtained in the first phase
        states_in_first_phase = []
        for state in states:
            states_in_first_phase.append(xp.copy(state))

        ### Second Phase ###
        beta = 1.0
        for iteration in range(100):
            # compute gradient of the loss function with respect to x
            output_state = states[-1]
            variable = chainer.Variable(output_state)
            loss = cf.mean_squared_error(variable, y)
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
            eps = 0.001
            for state, mu in zip(states, mu_array):
                new_state = clip(state + eps * mu)
                state[...] = new_state

        output_state = states[-1]
        loss = xp.sum((output_state - y)**2)
        print(output_state)
        print(y)
        print(loss)

        # store states obtained in the first phase
        states_in_second_phase = []
        for state in states:
            states_in_second_phase.append(xp.copy(state))

        # update weights
        # mu_array = []
        # assert len(states) == len(weight_pairs)
        # for (bottom_state, state,
        #      top_state), (Wf, Wb), state_first, state_second in zip(
        #          neighbor_state_pairs, weight_pairs, states_in_first_phase,
        #          states_in_second_phase):
        #     vWf = chainer.Variable(Wf)
        #     mu = cf.sum(
        #         cf.connection.linear.linear(rho(bottom_state), vWf) *
        #         (state_second - state_first))
        #     mu.backward()
        #     Wf[...] -= 0.01 * vWf.grad

        #     if Wb is None:
        #         continue

        #     vWb = chainer.Variable(Wb)
        #     mu = cf.sum(
        #         cf.connection.linear.linear(rho(top_state), vWb) *
        #         (state_second - state_first))
        #     mu.backward()
        #     Wb[...] -= 0.01 * vWb.grad

        # evaluate
        # for iteration in range(200):
        #     # compute vector field
        #     mu_array = []
        #     assert len(states) == len(weight_pairs)
        #     for (bottom_state, state, top_state), (Wf, Wb) in zip(
        #             neighbor_state_pairs, weight_pairs):
        #         # output layer
        #         if Wb is None:
        #             assert top_state is None
        #             mu = rho(bottom_state).dot(Wf) - state
        #             mu_array.append(mu)
        #             continue

        #         # other layers
        #         mu = rho(bottom_state).dot(Wf) + rho(top_state).dot(
        #             Wb) - state
        #         mu_array.append(mu)

        #     # update states
        #     eps = 0.001
        #     for state, mu in zip(states, mu_array):
        #         new_state = rho(state - eps * mu)
        #         state[...] = new_state

        # output_state = states[-1]
        # loss = xp.sum((output_state - y)**2)
        # print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--batchsize", "-bs", type=int, default=20)
    parser.add_argument("--ndim-hidden", "-ndim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", "-layers", type=int, default=1)
    args = parser.parse_args()
    main()
