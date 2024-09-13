import timeit 

import random

import numba

import minitorch

from .run_fast_tensor import datasets, FastTensorBackend 
from .run_fast_tensor import FastTrain

if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 100
    data = datasets["Xor"](PTS)
    run_times = 10
    print("cpu", timeit.timeit("FastTrain(HIDDEN, backend=FastTensorBackend, max_epochs=10).train(data, 0.05)", number=run_times) / run_times)
    print("gpu", timeit.timeit("FastTrain(HIDDEN, backend=FastTensorBackend, max_epochs=10).train(data, 0.05)", number=run_times) / run_times)



