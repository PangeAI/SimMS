import warnings
from typing import Type
import numpy as np
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


@memory.cache
def get_expected_score(
    method: Type,
    references: tuple,
    queries: tuple,
    **kwargs,
) -> np.ndarray:
    from matchms.similarity import CosineGreedy

    return method(**kwargs).matrix(
        references=references,
        queries=queries,
    )
