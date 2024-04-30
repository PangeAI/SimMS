import os, warnings
from contextlib import contextmanager

# Patch cuda.pinned and as_cuda_array before https://github.com/numba/numba/pull/9458 gets merged
if os.getenv("NUMBA_ENABLE_CUDASIM") == "1":
    from numba import cuda

    @contextmanager
    def fake_cuda_pinned(*arylist):
        yield

    def fake_as_cuda_array(obj, sync=True):
        return obj

    cuda.pinned = fake_cuda_pinned
    cuda.as_cuda_array = fake_as_cuda_array

# Same warnings are given once
warnings.simplefilter('default', Warning)
