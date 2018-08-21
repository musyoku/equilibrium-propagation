import argparse

import cupy as cp
import numpy as np
from chainer.backends import cuda

from model import Model


def main():
    # gpu/cpu
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    # model
    model = Model(
        ndim_hidden=args.ndim_hidden, num_hidden_layers=args.num_hidden_layers)
    if using_gpu:
        model.to_gpu()

    x = np.random.normal(size=(args.batchsize, 784))
    y = np.random.normal(size=(args.batchsize, 10))

    # training loop
    states = model.generate_initial_states(args.batchsize)
    print(states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--batchsize", "-bs", type=int, default=512)
    parser.add_argument("--ndim-hidden", "-ndim", type=int, default=512)
    parser.add_argument("--num-hidden-layers", "-layers", type=int, default=1)
    args = parser.parse_args()
    main()
