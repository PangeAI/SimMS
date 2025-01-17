import numpy as np
from matchms import calculate_scores
from simms.similarity import CudaCosineGreedy
from ..utils import get_expected_cosine_greedy_score


def test_sparse_calculate_scores(
    gnps: list,
):
    batch_size = 512
    match_limit = 1024
    sparse_threshold = 0.75

    r = gnps[:batch_size]
    q = gnps[:batch_size]

    expected_score = get_expected_cosine_greedy_score(r, q)
    kernel = CudaCosineGreedy(
        batch_size=batch_size,
        match_limit=match_limit,
        sparse_threshold=sparse_threshold,
        verbose=False,
    )
    result = calculate_scores(
        references=r,
        queries=q,
        similarity_function=kernel,
        array_type="sparse",
    )

    score = result.to_array("CudaCosineGreedy_sparse_score")
    matches = result.to_array("CudaCosineGreedy_sparse_matches")
    overflow = result.to_array("CudaCosineGreedy_sparse_overflow")

    is_discarded = expected_score["score"] < sparse_threshold
    equals = np.isclose(expected_score["score"], score, atol=0.001)
    match_equals = np.isclose(expected_score["matches"], matches)

    equals_except_overflows = equals | is_discarded | overflow
    match_equals_except_overflows = match_equals | is_discarded | overflow

    accuracy_rate = equals_except_overflows.mean()
    inaccuracy_num = (1 - equals_except_overflows).sum()
    match_accuracy_rate = match_equals_except_overflows.mean()
    match_inaccuracy_num = (1 - match_equals_except_overflows).sum()
    overflow_rate = overflow.mean()
    overflow_num = overflow.sum()

    report = f"""
    accuracy={accuracy_rate:.7f}, num incorrect {inaccuracy_num}
    match_acc={match_accuracy_rate:.7f}, num incorrect {match_inaccuracy_num}
    overflow={overflow_rate:.7f}, num overflow {overflow_num}
    """
    assert accuracy_rate > 0.999 and match_accuracy_rate > 0.999, f"{report}"
