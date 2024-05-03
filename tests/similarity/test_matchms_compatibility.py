from typing import List

import numpy as np
import pytest
from matchms import Scores, Spectrum, calculate_scores
from matchms.similarity import (BaseSimilarity, CosineGreedy,
                                FingerprintSimilarity, ModifiedCosine)
from cudams.similarity import CudaCosineGreedy, CudaFingerprintSimilarity, \
                                CudaModifiedCosine
from ..builder_Spectrum import SpectrumBuilder
from joblib import Memory


memory = Memory(location="cache")

def equality_function(prefix:str):
    def equality(scores: Scores, scores_cu: Scores):
        score = scores[f"{prefix}_score"]
        score_cu = scores_cu[f"Cuda{prefix}_score"]
        matches = scores[f"{prefix}_matches"]
        matches_cu = scores_cu[f"Cuda{prefix}_matches"]
        not_ovfl = 1 - scores_cu[f"Cuda{prefix}_overflow"]

        # We allow only overflowed values to be different (don't count toward acc)
        acc = np.isclose(matches * not_ovfl, matches_cu * not_ovfl, equal_nan=True)
        assert acc.mean() == 1

        acc = np.isclose(score * not_ovfl, score_cu * not_ovfl, equal_nan=True)
        assert acc.mean() == 1

        # We allow only few overflows
        assert not_ovfl.mean() >= 0.99
    return equality

def equality_function_fingerprint(
    scores: Scores,
    scores_cu: Scores,
):
    assert np.allclose(scores, scores_cu, equal_nan=True)


@pytest.mark.parametrize(
    "similarity_function, args, cuda_similarity_function, cu_args, equality_function",
    [

        (CosineGreedy, (), CudaCosineGreedy, (), equality_function('CosineGreedy')),
        (ModifiedCosine, (), CudaModifiedCosine, (), equality_function('ModifiedCosine')),
        (
            FingerprintSimilarity,
            ("jaccard",),
            CudaFingerprintSimilarity,
            ("jaccard",),
            equality_function_fingerprint,
        ),
        (
            FingerprintSimilarity,
            ("cosine",),
            CudaFingerprintSimilarity,
            ("cosine",),
            equality_function_fingerprint,
        ),
        (
            FingerprintSimilarity,
            ("dice",),
            CudaFingerprintSimilarity,
            ("dice",),
            equality_function_fingerprint,
        ),
    ],
)
def test_compatibility(
    gnps_with_fingerprint: List[Spectrum],
    similarity_function: BaseSimilarity,
    args: tuple,
    cuda_similarity_function: BaseSimilarity,
    cu_args: tuple,
    equality_function: callable,
):
    references, queries = gnps_with_fingerprint[:256], gnps_with_fingerprint[:256]
    similarity_function = similarity_function(*args)
    scores = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=similarity_function,
        is_symmetric=True,
    ).to_array()
    cuda_similarity_function = cuda_similarity_function(*cu_args)

    scores_cu = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=cuda_similarity_function,
    ).to_array()

    equality_function(scores, scores_cu)
