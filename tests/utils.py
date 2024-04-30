import numpy as np
import warnings
import pytest
from joblib import Memory

memory = Memory(location="cache")

@memory.cache
def get_expected_cosine_greedy_score(
        references: tuple, 
        queries: tuple,
        **kwargs,
    ) -> np.ndarray:
    from matchms.similarity import CosineGreedy
    return CosineGreedy(**kwargs).matrix(
        references=references,
        queries=queries,
    )