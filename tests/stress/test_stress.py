from typing import Type
import matchms
import numpy as np
import pytest
from joblib import Memory
from matchms.filtering import reduce_to_number_of_peaks
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.similarity import CosineGreedy, ModifiedCosine
from simms.similarity import CudaCosineGreedy, CudaModifiedCosine
from ..utils import get_expected_cosine_greedy_score, get_expected_score
from sparsestack import StackedSparseArray

memory = Memory("cache", verbose=False)


@memory.cache
def trimmed_spectra(gnps, n_max_peaks):
    spectra = [reduce_to_number_of_peaks(sp, n_max=n_max_peaks) for sp in gnps]
    spectra = [sp for sp in spectra if sp is not None]
    return spectra

@pytest.mark.parametrize(
    "array_type",
    [
        'sparse',
        'numpy'
    ]
)
@pytest.mark.parametrize(
    "method, method_cu",
    [
        [CosineGreedy, CudaCosineGreedy],
        [ModifiedCosine, CudaModifiedCosine],
    ],
)
@pytest.mark.parametrize(
    "reference_size, query_size",
    [
        [64,128], # large data
        [16, 32], # small data
    ]
)
@pytest.mark.parametrize(
    "tolerance, batch_size, n_max_peaks, match_limit, mz_power, intensity_power",
    [
        (0.1, 256, 512, 512, 0, 1,),  # batch smaller than data
        (0.1, 15, 512, 512, 0, 1,),  # batch larger than data

        # (0.1, 31, 32, 31, 1, 0,),  # Representative case 1
        # (0.01, 65, 121, 63, 1, 1,),  # Representative case 2
        # (0.01, 65, 121, 63, 1, 1,),  # Representative case 2
        # (1e-6, 118, 500, 127, 2, 2),  # Representative case 3
        # (0.1, 200, 1200, 257, 1, 0),  # Representative case 4
        # (0.01, 499, 300, 511, 1, 0),  # Representative case 5
        # (1e-6, 80, 1200, 1023, 2, 1),  # Representative case 6
        # (0.1, 18, 2047, 2049, 0, 2),  # Representative case 7
        # (0.01, 65, 2049, 2047, 1, 2),  # Representative case 8
    ],
)
def test_stress(
    method: Type[BaseSimilarity],
    method_cu: Type[BaseSimilarity],
    gnps: list,
    reference_size: int,
    query_size: int,
    tolerance: float,
    batch_size: int,
    n_max_peaks: int,
    match_limit: int,
    mz_power: float,
    intensity_power: float,
    array_type:str,
):
    spectra = trimmed_spectra(tuple(gnps[:reference_size+query_size]), n_max_peaks)
    references, queries = spectra[:reference_size], spectra[reference_size:]

    # This is in a method since we use caching
    result_target = get_expected_score(
        method,
        references,
        queries,
        tolerance=tolerance,
        mz_power=mz_power,
        intensity_power=intensity_power,
    )

    kernel:BaseSimilarity = method_cu(
        mz_power=mz_power,
        intensity_power=intensity_power,
        tolerance=tolerance,
        batch_size=batch_size,
        n_max_peaks=n_max_peaks,
        match_limit=match_limit,
        sparse_threshold=.1,
        verbose=False,
    )
    result = kernel.matrix(references, queries, array_type=array_type)

    if array_type == 'sparse':
        assert isinstance(result, StackedSparseArray)
        result: StackedSparseArray
        result = result.to_array()
        
        overflow = result['sparse_overflow']
        not_filtered = result_target['score'] > .1 # Not matching sub-threshold scores
        mask = (~overflow) & not_filtered
        equals = np.isclose(result_target['score'], result['sparse_score'], atol=1e-3) | mask
        match_equals = np.isclose(result_target['matches'], result['sparse_matches']) | mask
        equals_except_overflows = equals | overflow
        match_equals_except_overflows = match_equals | overflow
        overflow_num = overflow.sum()
        overflow_rate = overflow.sum()
    else:
        assert isinstance(result, np.ndarray)
        equals = np.isclose(result_target["score"], result["score"], atol=1e-3)
        match_equals = np.isclose(result_target["matches"], result["matches"])
        equals_except_overflows = equals | result["overflow"]
        match_equals_except_overflows = match_equals | result["overflow"]
        overflow_rate = result["overflow"].mean()
        overflow_num = result["overflow"].sum()

    # Calculate accuracy rates and overflow statistics
    accuracy_rate = equals_except_overflows.mean()
    inaccuracy_num = (1 - equals_except_overflows).sum()
    match_accuracy_rate = match_equals_except_overflows.mean()
    match_inaccuracy_num = (1 - match_equals_except_overflows).sum()

    # Prepare error and warning messages
    errors = []
    warns = []
    if accuracy_rate < 0.99:
        errors.append(f"accuracy={accuracy_rate:.7f} # {inaccuracy_num}")
    if match_accuracy_rate < 0.99:
        errors.append(f"match_acc={match_accuracy_rate:.7f} # {match_inaccuracy_num}")
    if overflow_rate > 0:
        warns.append(f"overflow={overflow_rate:.7f} # {overflow_num}")

    assert not errors, f"ERR: {errors}, WARN: {warns}"
