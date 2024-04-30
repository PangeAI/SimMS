from pathlib import Path

import numpy as np
from matchms import calculate_scores
from matchms.filtering import (default_filters, normalize_intensities,
                               reduce_to_number_of_peaks)
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy
from numba import cuda

from cudams.similarity import CudaCosineGreedy
from cudams.utils import download


def test_tutorial_pesticide():
    assert cuda.is_available()
    from matchms import calculate_scores
    from matchms.importing import load_from_mgf
    from cudams.similarity import CudaCosineGreedy
    pest_file = download("pesticides.mgf")

    references = list(load_from_mgf(pest_file))
    queries = list(load_from_mgf(pest_file))

    kernel = CudaCosineGreedy()

    scores = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=kernel,
    )

    best_matches = scores.scores_by_query(queries[42], 'CudaCosineGreedy_score', sort=True)

    scores = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=CosineGreedy(),
    )
    best_matches_cu = scores.scores_by_query(queries[42], 'CosineGreedy_score', sort=True)

    for a, b in zip(best_matches[:20], best_matches_cu[:20]):
        reference, (score_a, matches_a, overflows_a) = a
        reference, (score_b, matches_b) = b
        assert np.isclose(score_a, score_b) and np.isclose(matches_a, matches_b)