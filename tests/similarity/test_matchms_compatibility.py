from typing import List
import numpy as np
import pytest
from joblib import Memory
from matchms import Scores, Spectrum, calculate_scores
from matchms.similarity import (
    BaseSimilarity,
    CosineGreedy,
    FingerprintSimilarity,
    ModifiedCosine,
)
from cudams.similarity import (
    CudaCosineGreedy,
    CudaFingerprintSimilarity,
    CudaModifiedCosine,
)
from cudams.utils import get_correct_scores
from ..builder_Spectrum import SpectrumBuilder


def equality_function(prefix: str):
    def equality(scores: Scores, scores_cu: Scores):
        score = scores[f"score"]
        matches = scores[f"matches"]

        score_cu = scores_cu[f"Cuda{prefix}_score"]
        matches_cu = scores_cu[f"Cuda{prefix}_matches"]
        not_ovfl = 1 - scores_cu[f"Cuda{prefix}_overflow"]

        # We allow only overflowed values to be different (don't count toward acc)
        acc = np.isclose(matches * not_ovfl, matches_cu * not_ovfl, equal_nan=True)
        assert acc.mean() >= .99

        acc = np.isclose(score * not_ovfl, score_cu * not_ovfl, equal_nan=True)
        assert acc.mean() >= .99

        # We allow only few overflows
        assert not_ovfl.mean() >= 0.99

    return equality


def equality_function_fingerprint(
    scores: Scores,
    scores_cu: Scores,
):
    acc = np.isclose(scores, scores_cu, equal_nan=True)
    assert acc.mean() >= .99


@pytest.mark.parametrize(
    "SimilarityClass, args, CudaSimilarityClass, cu_args, equality_function",
    [
        (CosineGreedy, dict(), CudaCosineGreedy, dict(), equality_function("CosineGreedy")),
        (
            ModifiedCosine,
            dict(),
            CudaModifiedCosine,
            dict(),
            equality_function("ModifiedCosine"),
        ),
        (
            FingerprintSimilarity,
            dict(similarity_measure="jaccard",),
            CudaFingerprintSimilarity,
            dict(similarity_measure="jaccard",),
            equality_function_fingerprint,
        ),
        (
            FingerprintSimilarity,
            dict(similarity_measure="cosine",),
            CudaFingerprintSimilarity,
            dict(similarity_measure="cosine",),
            equality_function_fingerprint,
        ),
        (
            FingerprintSimilarity,
            dict(similarity_measure="dice",),
            CudaFingerprintSimilarity,
            dict(similarity_measure="cosine",),
            equality_function_fingerprint,
        ),
    ],
)
def test_compatibility(
    gnps_with_fingerprint: List[Spectrum],
    SimilarityClass: BaseSimilarity,
    args: dict,
    CudaSimilarityClass: BaseSimilarity,
    cu_args: dict,
    equality_function: callable,
):
    references, queries = gnps_with_fingerprint[:256], gnps_with_fingerprint[:256]

    scores = get_correct_scores(
        references=references, queries=queries, similarity_class=SimilarityClass, **args
    )

    cuda_kernel = CudaSimilarityClass(**cu_args, batch_size=max(len(references), len(queries)))
    scores_cu = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=cuda_kernel,
    ).to_array()

    equality_function(scores, scores_cu)
