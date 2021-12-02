import numpy as np
from numpy.linalg import norm
import bottleneck as bn

"""
   TODO:
   implement Mantel test to compare distance matrices 
   https://github.com/jwcarr/mantel
   https://en.wikipedia.org/wiki/Mantel_test
   https://users.aalto.fi/~eglerean/permutations.html
   
   
"""


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    The resulting similarity value will range from -1 (opposite) to 1 (same).

    Parameters
    ----------
    a : np.ndarray
        first input vector, stack of vectors (n_samples, n_vectors)
    b : np.ndarray
        other input vector,, stack of vectors (n_samples, n_vectors)

    Returns
    -------
    float
        cosine similarity

    Raises
    ------
    ValueError
        if inputs do not have the same size
    
    References
    ----------
    https://en.wikipedia.org/wiki/Cosine_similarity
    https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    if not np.array_equal(a.shape, b.shape):
        raise ValueError("inputs should have the same shape")

    return a @ b.T / np.outer(norm(a, axis=1), norm(b, axis=1))


def pairwise_euclidian(A: np.ndarray, B: np.ndarray):
    """
    pairwise_euclidian calculate euclidian distance aon all the pairs of row from the
    input matrices

    Based on the fact that:
        (a - b)^2 = a^2 + b^2 - 2ab

    Parameters
    ----------
    A : np.ndarray
        [description]
    B : np.ndarray
        [description]

    Returns
    -------
    distance matrix 
        [description]

    Reference
    ---------
    https://codereview.stackexchange.com/a/77270    
    """

    d = bn.nansum((A ** 2), axis=-1)[:, np.newaxis] + bn.nansum(B ** 2, axis=-1)
    d -= 2 * np.squeeze(A @ B.T)
    return np.sqrt(d)
