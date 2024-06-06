from cudams.similarity import CudaCosineGreedy
from cudams.utils import get_correct_scores
from matchms import Spectrum
from matchms.similarity import CosineGreedy
from typing import List
import pytest
import numpy as np



def test_unpacking_sparse_scores_works(
    gnps: List[Spectrum],
):
    np.random.seed(42)
    references = np.random.choice(gnps, 100)
    queries = np.random.choice(gnps, 150)

    scores = CudaCosineGreedy(
        sparse_threshold=.75
    ).matrix(references,queries, array_type='sparse')

    # CASE 1
    assert np.allclose(scores.to_array().shape, [100,150])
    assert all(scores.data['sparse_score'] >= .75) # ref ID, query ID, 
    assert len(scores.data['sparse_matches']) == len(scores.data['sparse_score']) # ref ID, query ID, matches
