from typing import Type
import matchms
import numpy as np
import pytest
from joblib import Memory
from matchms.filtering import reduce_to_number_of_peaks
from matchms.similarity import CosineGreedy, ModifiedCosine
from cudams.similarity import CudaCosineGreedy, CudaModifiedCosine
from ..utils import get_expected_cosine_greedy_score, get_expected_score


memory = Memory('cache', verbose=False)

@memory.cache
def trimmed_spectra(gnps, n_max_peaks):
    spectra = [reduce_to_number_of_peaks(sp, n_max=n_max_peaks) for sp in gnps]
    spectra = [sp for sp in spectra if sp is not None]
    return spectra

@pytest.mark.parametrize('method, method_cu', [
    [ModifiedCosine, CudaModifiedCosine],
    [CosineGreedy, CudaCosineGreedy],
])
@pytest.mark.parametrize(
    'tolerance, batch_size, n_max_peaks, match_limit, mz_power, intensity_power', 
    [
        (0.1, 31, 32, 31, 1, 0),    # Representative case 1
        (0.01, 65, 121, 63, 1, 1),  # Representative case 2
        (1e-6, 118, 500, 127, 2, 2),  # Representative case 3
        (0.1, 200, 1200, 257, 1, 0),  # Representative case 4
        (0.01, 499, 300, 511, 1, 0),  # Representative case 5
        (1e-6, 80, 1200, 1023, 2, 1),  # Representative case 6
        (0.1, 18, 2047, 2049, 0, 2),  # Representative case 7
        (0.01, 65, 2049, 2047, 1, 2),  # Representative case 8
    ]
)
def test_stress(
    method: Type, 
    method_cu: Type,
    gnps: list,
    tolerance: float,
    batch_size: int,
    n_max_peaks: int,
    match_limit: int,
    mz_power: float,
    intensity_power: float,
):
    spectra = trimmed_spectra(tuple(gnps[:batch_size * 2]), n_max_peaks)
    references, queries = spectra[:batch_size], spectra[batch_size:]

    # This is in a method since we use caching
    expected_score = get_expected_score(
        method, references, queries, 
        tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power,
    )

    kernel = method_cu(
        mz_power=mz_power,
        intensity_power=intensity_power,
        tolerance=tolerance,
        batch_size=batch_size, 
        n_max_peaks=n_max_peaks,
        match_limit=match_limit, 
        verbose=False
    )
    result = kernel.matrix(references, queries)

    # Check similarity scores and matches
    equals = np.isclose(expected_score['score'], result['score'], atol=1e-3)
    match_equals = np.isclose(expected_score['matches'], result['matches'])
    equals_except_overflows = equals | result['overflow']
    match_equals_except_overflows = match_equals | result['overflow']

    # Calculate accuracy rates and overflow statistics
    accuracy_rate = equals_except_overflows.mean()
    inaccuracy_num = (1 - equals_except_overflows).sum()
    match_accuracy_rate = match_equals_except_overflows.mean()
    match_inaccuracy_num = (1 - match_equals_except_overflows).sum()
    overflow_rate = result['overflow'].mean()
    overflow_num = result['overflow'].sum()
    
    # Prepare error and warning messages
    errors = []
    warns = []
    if accuracy_rate < 1:
        errors.append(f'accuracy={accuracy_rate:.7f} # {inaccuracy_num}')
    if match_accuracy_rate < 1:
        errors.append(f'match_acc={match_accuracy_rate:.7f} # {match_inaccuracy_num}')
    if overflow_rate > 0:
        warns.append(f'overflow={overflow_rate:.7f} # {overflow_num}')
    
    assert not errors, f"ERR: {errors}, WARN: {warns}"
