import timeit 
import warnings

import numba
from numba.core.errors import NumbaPerformanceWarning

import minitorch
from run_fast_tensor import datasets, FastTensorBackend, FastTrain

# Suppress Numba performance warnings
warnings.simplefilter("ignore", NumbaPerformanceWarning)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 100
    data = datasets["Circle"](PTS)
    run_times = 10
    print("cpu", timeit.timeit("FastTrain(HIDDEN, backend=FastTensorBackend).train(data, 0.05, max_epochs=100)", number=run_times, globals=globals()) / run_times / 100)
    if numba.cuda.is_available():
        print("gpu", timeit.timeit("FastTrain(HIDDEN, backend=GPUBackend).train(data, 0.05, max_epochs=100)", number=run_times, globals=globals()) / run_times / 100)


