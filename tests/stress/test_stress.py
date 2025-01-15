from typing import Type
import numpy as np
import pytest
from joblib import Memory
from matchms.filtering import reduce_to_number_of_peaks
from matchms.similarity import CosineGreedy, ModifiedCosine
from matchms.similarity.BaseSimilarity import BaseSimilarity
from sparsestack import StackedSparseArray
from simms.similarity import CudaCosineGreedy, CudaModifiedCosine


memory = Memory("cache", verbose=False)


def trimmed_spectra(gnps, n_max_peaks):
    spectra = [reduce_to_number_of_peaks(sp, n_max=n_max_peaks) for sp in gnps]
    spectra = [sp for sp in spectra if sp is not None]
    return spectra


@pytest.mark.parametrize(
    "array_type",
    [
        "sparse",
        "numpy",
    ],
)
@pytest.mark.parametrize(
    "Similarity, CudaSimilarity",
    [
        [CosineGreedy, CudaCosineGreedy],
        [ModifiedCosine, CudaModifiedCosine],
    ],
)
@pytest.mark.parametrize(
    "reference_size, query_size",
    [
        [64, 128],  # large data
        [16, 32],  # small data
    ],
)
@pytest.mark.parametrize(
    "tolerance, batch_size, n_max_peaks, match_limit, mz_power, intensity_power",
    [
        (
            0.1,
            64,
            1024,
            1024,
            0,
            1,
        ),  # batch smaller than data
    ],
)
def test_stress(
    Similarity: Type[BaseSimilarity],
    CudaSimilarity: Type[BaseSimilarity],
    gnps: list,
    reference_size: int,
    query_size: int,
    tolerance: float,
    batch_size: int,
    n_max_peaks: int,
    match_limit: int,
    mz_power: float,
    intensity_power: float,
    array_type: str,
):
    spectra = trimmed_spectra(tuple(gnps[: reference_size + query_size]), n_max_peaks)
    references, queries = spectra[:reference_size], spectra[reference_size:]

    kernel: BaseSimilarity = Similarity(
        tolerance=tolerance,
        mz_power=mz_power,
        intensity_power=intensity_power,
    )
    target = kernel.matrix(references, queries)

    kernel: BaseSimilarity = CudaSimilarity(
        mz_power=mz_power,
        intensity_power=intensity_power,
        tolerance=tolerance,
        batch_size=batch_size,
        n_max_peaks=n_max_peaks,
        match_limit=match_limit,
        sparse_threshold=0.1,
        verbose=False,
    )

    if array_type == "sparse":
        result: StackedSparseArray = kernel.matrix(
            references, queries, array_type=array_type
        )
        assert isinstance(result, StackedSparseArray)
        result = result.to_array()

        score = result["sparse_score"]
        matches = result["sparse_matches"]
        overflow = result["sparse_overflow"]

        mask = target["score"] > 0.1

        equals = np.isclose(target["score"] * mask, score * mask, atol=1e-3)
        assert equals.mean() > 0.999
        # not_filtered = result_target['score'] > .1 # Not matching sub-threshold scores
        # mask = (~overflow) & not_filtered
        # equals = np.isclose(result_target['score'], result['sparse_score'], atol=1e-3) | mask
        match_equals = np.isclose(matches, result["sparse_matches"]) | mask
        equals_except_overflows = equals | overflow
        match_equals_except_overflows = match_equals | overflow
        overflow_rate = overflow.mean()
        overflow_num = overflow.sum()
    else:
        result: np.ndarray = kernel.matrix(references, queries, array_type=array_type)
        assert isinstance(result, np.ndarray)
        equals = np.isclose(target["score"], result["score"], atol=1e-3)
        match_equals = np.isclose(target["matches"], result["matches"])
        equals_except_overflows = equals | result["overflow"]
        match_equals_except_overflows = match_equals | result["overflow"]
        overflow_rate = result["overflow"].mean()
        overflow_num = result["overflow"].sum()

    # Calculate accuracy rates and overflow statistics
    accuracy_rate = equals_except_overflows.mean()
    inaccuracy_num = (1 - equals_except_overflows).sum()
    match_accuracy_rate = match_equals_except_overflows.mean()
    match_inaccuracy_num = (1 - match_equals_except_overflows).sum()

    report = f"""
    accuracy={accuracy_rate:.7f}, num incorrect {inaccuracy_num}
    match_acc={match_accuracy_rate:.7f}, num incorrect {match_inaccuracy_num}
    overflow={overflow_rate:.7f}, num overflow {overflow_num}
    """
    assert accuracy_rate > 0.999, report
