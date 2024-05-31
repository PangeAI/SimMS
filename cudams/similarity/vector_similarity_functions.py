import math
import numba
import numpy as np
import torch
from numba import cuda, types
from torch import Tensor


device = "cuda" if torch.cuda.is_available() else "cpu"


def jaccard_similarity_matrix(
    references: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    """Returns matrix of jaccard indices between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """

    # We know references and queries have same number of elements (bits)
    refs = (
        torch.from_numpy(np.array(references, copy=False)).to(device).float()
    )  # Shape R, N
    ques = (
        torch.from_numpy(np.array(queries, copy=False)).to(device).float()
    )  # Shape Q, N

    intersection = refs @ ques.T  # Shape R, Q, all intersection rows are summed
    union = refs.sum(1, keepdim=True) + ques.sum(1, keepdim=True).T  # R, Q
    union -= intersection

    jaccard = intersection.div(union).nan_to_num()
    return jaccard.cpu().numpy()


def dice_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of dice similarity scores between all-vs-all vectors of references
    and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    # size1 = references.shape[0]
    # size2 = queries.shape[0]
    # scores = np.zeros((size1, size2))
    # for i in range(size1):
    #     for j in range(size2):
    #         scores[i, j] = dice_similarity(references[i, :], queries[j, :])
    # return scores

    # u_and_v = np.bitwise_and(u != 0, v != 0)
    # u_abs_and_v_abs = np.abs(u).sum() + np.abs(v).sum()
    # dice_score = 0
    # if u_abs_and_v_abs != 0:
    #     dice_score = 2.0 * np.float64(u_and_v.sum()) / np.float64(u_abs_and_v_abs)
    # return dice_score

    refs = torch.from_numpy(np.array(references)).to(device).float()  # Shape R, N
    ques = torch.from_numpy(np.array(queries)).to(device).float()  # Shape Q, N

    intersection = refs @ ques.T  # Shape R, Q, all intersection rows are summed
    union = refs.sum(1, keepdim=True).abs() + ques.sum(1, keepdim=True).abs().T  # R, Q

    dice = 2 * intersection.div(union).nan_to_num()
    return dice.cpu().numpy()


def cosine_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of cosine similarity scores between all-vs-all vectors of
    references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """

    refs = torch.from_numpy(np.array(references)).to(device).float()  # R,N
    ques = torch.from_numpy(np.array(queries)).to(device).float()  # Q,N
    score = refs @ ques.T  # R, Q
    norm = refs.pow(2).sum(1, keepdim=True) @ ques.pow(2).sum(1, keepdim=True).T  # R, Q
    score = score.div(norm.sqrt()).nan_to_num(0)
    return score.cpu().numpy()
