from scipy import sparse
import numpy as np


def var_s(a, axis=-1, ddof=1):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    n = a.shape[axis]
    a_squared = a.copy()
    a_squared.data **= 2
    sig = a_squared.mean(axis) - np.square(a.mean(axis))
    sig = (sig * n) / (n - ddof)
    return sig


def std_s(a, axis=-1, ddof=1):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(var_s(a, axis=axis, ddof=ddof))


def corr_stack_s(a: sparse.csr_matrix, b: sparse.csr_matrix):
    """
    calculate the cross correlation matrix between all 
    the possible pairs of row between each matrices.
    keep most of the things sparse to have a fast calculation

    Leverage the fact that:
                - corr(x,y) = cov(x,y) / (sigma(x) sigma(y))
    
    Parameters
    ----------
    a : sparse.csr_matrix
        first input
    b : sparse.csr_matrix
        second input

    Returns
    -------
    Correlation matrix

    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    https://stackoverflow.com/questions/19231268/correlation-coefficients-for-sparse-matrix-in-python
    """

    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    if not sparse.issparse(b):
        b = sparse.csr_matrix(b)

    n = a.shape[1]
    a_m = a.sum(1)
    b_m = b.sum(1)

    a_s = std_s(a, axis=1, ddof=1)
    b_s = std_s(b, axis=1, ddof=1)

    E_XY = a @ b.T
    E_X_E_Y = a_m.dot(b_m.T) / n

    # compute the covariance matrix
    Cov = (E_XY - E_X_E_Y) / (n - 1)

    # compute the correlation matrix
    Corr = Cov / (a_s.dot(b_s.T))

    return Corr


def overlap_s(a: sparse.csr_matrix, b: sparse.csr_matrix):

    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    if not sparse.issparse(b):
        b = sparse.csr_matrix(b)

    return a @ b.T


def jacquard_s(a: sparse.csr_matrix, b: sparse.csr_matrix):
    """
    Jacquard distance can be written as:
            (A ∩ B) / (A ∪ B)

    which can be rewriten as:
            (A ∩ B) / (A + B  - (A ∩ B) )
    Parameters
    ----------
    a : [type]
        first sparse matrix
    b : [type]
        second sparse matrix 
    """
    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    if not sparse.issparse(b):
        b = sparse.csr_matrix(b)

    a_sz = a.sum(1)
    b_sz = b.sum(1)

    a_inter_b = overlap_s(a, b)
    divisor = a_sz + b_sz.T

    return a_inter_b / (divisor - a_inter_b)
