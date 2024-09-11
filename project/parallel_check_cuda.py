from numba.cuda import jit

import minitorch

# MAP
print("MAP")
tmap = minitorch.cuda_ops.tensor_map(jit(device=True)(minitorch.operators.id))
out, a = minitorch.zeros((10,)), minitorch.zeros((10,))
tmap[1, 32](*out.tuple(), 10, *a.tuple())
print(tmap.parallel_diagnostics(level=3))

# ZIP
print("ZIP")
out, a, b = minitorch.zeros((10,)), minitorch.zeros((10,)), minitorch.zeros((10,))
tzip = minitorch.cuda_ops.tensor_zip(jit(device=True)(minitorch.operators.eq))
print(out.tuple())
tzip[1, 32](*out.tuple(), 10, *a.tuple(), *b.tuple())
print(tzip.parallel_diagnostics(level=3))

# REDUCE
print("REDUCE")
out, a = minitorch.zeros((1,)), minitorch.zeros((10,))
treduce = minitorch.cuda_ops.tensor_reduce(jit(device=True)(minitorch.operators.add))

treduce[1, 1024](*out.tuple(), 1, *a.tuple(), 0, 0.0)
print(treduce.parallel_diagnostics(level=3))


# MM
# print("MATRIX MULTIPLY")
# out, a, b = (
#     minitorch.zeros((1, 10, 10)),
#     minitorch.zeros((1, 10, 20)),
#     minitorch.zeros((1, 20, 10)),
# )
# tmm = minitorch.fast_ops.tensor_matrix_multiply

# tmm(*out.tuple(), *a.tuple(), *b.tuple())
# print(tmm.parallel_diagnostics(level=3))
