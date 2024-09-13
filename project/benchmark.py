import timeit 
import numba

import minitorch
from run_fast_tensor import datasets, FastTensorBackend, FastTrain

if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 100
    data = datasets["Circle"](PTS)
    run_times = 10
    print("cpu", timeit.timeit("FastTrain(HIDDEN, backend=FastTensorBackend).train(data, 0.05, max_epochs=10)", number=run_times, globals=globals()) / run_times)
    if numba.cuda.is_available():
        print("gpu", timeit.timeit("FastTrain(HIDDEN, backend=GPUBackend).train(data, 0.05, max_epochs=10)", number=run_times, globals=globals()) / run_times)



