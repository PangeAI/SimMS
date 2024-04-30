from pathlib import Path
import pytest
import numpy as np
from cudams.similarity import CudaCosineGreedy


@pytest.mark.parametrize(
    'batch_size, match_limit, n_max_peaks, array_type, sparse_threshold', 
    [
        [512, 2048, 1024, 'numpy', 0,],
        [1024, 2048, 1024, 'numpy', 0,],
        [2048, 2048, 1024, 'numpy', 0,],
        # [1024, 512, 512, 'numpy', 0,],
        # [1024, 1024, 512, 'numpy', 0,],
        # [1024, 512, 1024, 'numpy', 0,],

        # [1024, 1024, 2048, 'numpy', 0,],
        # [1024, 2048, 1024, 'numpy', 0,],

        # [1024, 1024, 1024, 'numpy', 0,],
        # [1024, 1024, 1024, 'numpy', 0,],
        # [2048, 1024, 1024, 'numpy', 0,],
        # [1024, 1024, 1024, 'sparse', .75],
    ]
)
@pytest.mark.performance
def test_performance(
    gnps: list,
    batch_size: int,
    match_limit: int,
    n_max_peaks: int,
    array_type: str,
    sparse_threshold: float,
):
    kernel = CudaCosineGreedy(batch_size=batch_size,
                              n_max_peaks=n_max_peaks,
                              match_limit=match_limit, 
                              sparse_threshold=sparse_threshold,
                              verbose=False)
    # Warm-up
    # kernel.matrix(gnps[:4], gnps[:4])
    kernel.matrix(
        gnps[:batch_size],
        gnps[:batch_size],
        array_type=array_type,
    )
    print(f"\n=> PERF:  {kernel.kernel_time:.4f}s @ Bs:{batch_size}, ml:{match_limit}, np:{n_max_peaks} at:{array_type}, sp:{sparse_threshold}, \n")