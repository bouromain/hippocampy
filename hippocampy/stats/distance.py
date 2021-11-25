import numpy as np
from numpy.linalg import norm


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

