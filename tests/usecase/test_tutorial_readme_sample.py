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

def readme_fn():
    from matchms import calculate_scores
    from matchms.importing import load_from_mgf
    from cudams.utils import download
    from cudams.similarity import CudaCosineGreedy, \
                                CudaModifiedCosine, \
                                CudaFingerprintSimilarity

    sample_file = download('pesticides.mgf')
    references = list(load_from_mgf(sample_file))
    queries = list(load_from_mgf(sample_file))

    similarity_function = CudaCosineGreedy()

    scores = calculate_scores( 
    references=references,
    queries=queries,
    similarity_function=similarity_function, 
    )

    scores.scores_by_query(queries[42], 'CudaCosineGreedy_score', sort=True)
    return scores

def test_readme_not_fail():
    readme_fn()
    
def test_readme_correct():
    
    scores_cu = readme_fn()

    from matchms import calculate_scores
    from matchms.importing import load_from_mgf
    from matchms.similarity import CosineGreedy
    from cudams.utils import download

    sample_file = download('pesticides.mgf') # Download sample file
    references = list(load_from_mgf(sample_file)) # Read using MatchMS
    queries = list(load_from_mgf(sample_file))

    scores = calculate_scores(
        references=references, 
        queries=queries,
        similarity_function=CosineGreedy(),
    )
    assert np.isclose(scores._scores['CosineGreedy_score'], scores_cu._scores['CudaCosineGreedy_score']).all()