import os
import warnings
from contextlib import contextmanager
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from .__version__ import __version__


# Patch cuda.pinned and as_cuda_array before https://github.com/numba/numba/pull/9458 gets merged
if os.getenv("NUMBA_ENABLE_CUDASIM") == "1":
    from numba import cuda

    @contextmanager
    def fake_cuda_pinned(*arylist):
        yield

    def fake_as_cuda_array(obj, sync=True):
        return obj

    original_event = cuda.event

    def fake_event(*args, **kwargs):
        return original_event()

    cuda.pinned = fake_cuda_pinned
    cuda.as_cuda_array = fake_as_cuda_array
    cuda.event = fake_event

__author__ = "SimMS developers community"
__email__ = "tornikeonoprishvili@gmail.com"
__all__ = [
    "__version__",
    "similarity",
]
