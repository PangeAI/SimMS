import numpy as np
import torch
from simms.utils import get_device


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
    device = get_device()

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
    device = get_device()
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
    device = get_device()
    refs = torch.from_numpy(np.array(references)).to(device).float()  # R,N
    ques = torch.from_numpy(np.array(queries)).to(device).float()  # Q,N
    score = refs @ ques.T  # R, Q
    norm = refs.pow(2).sum(1, keepdim=True) @ ques.pow(2).sum(1, keepdim=True).T  # R, Q
    score = score.div(norm.sqrt()).nan_to_num(0)
    return score.cpu().numpy()
