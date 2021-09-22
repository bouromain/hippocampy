# Provide helper function, descriptive statistics and statistical tests
# for circular data
#
# References:
# [1]   Berens, Philipp. 2009. CircStat: A MATLAB Toolbox for Circular Statistics.
#       Journal of Statistical Software, Articles 31 (10): 1–21.
#
# [2]   Vallat, R. (2018). Pingouin: statistics in Python. Journal of Open Source
#       Software, 3(31), 1026, https://doi.org/10.21105/joss.01026
#
# [3]   Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in circular
#       statistics (Vol. 5). world scientific
#       https://www.google.co.uk/books/edition/Topics_in_Circular_Statistics/sKqWMGqQXQkC?hl=en&gbpv=1&pg=PP1&printsec=frontcover
#
# [4]   Rabin, J., Delon, J. & Gousseau, Y. Transportation Distances on the Circle.
#       J Math Imaging Vis 41, 147 (2011). https://doi.org/10.1007/s10851-011-0284-0
#       https://arxiv.org/pdf/0906.5499v2.pdf
#
# [5]   J. Rabin, J. Delon and Y. Gousseau, "Circular Earth Mover’s Distance for the
#       comparison of local features," 2008 19th International Conference on Pattern
#       Recognition, Tampa, FL, 2008, pp. 1-4, doi: 10.1109/ICPR.2008.4761372.
#       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.4116&rep=rep1&type=pdf

import numpy as np
import bottleneck as bn
from numpy import pi
from scipy.stats import norm
from scipy.optimize import fminbound
from math import erf

########################################################################
## Helpers
########################################################################
def isradians(x):
    """
    verify if input vector is in radians and return warning otherwise
    """
    if all(x >= -pi) & all(x <= pi):
        return 1
    elif all(x >= 0) & all(x <= 2 * pi):
        return 2
    else:
        # print("Warning, radians not in [-pi,pi] or [0,2pi] range")
        return 0


def wrap(x, rangeVal=1):
    """
    wrap radian values between range [-pi,pi] (range = 1) or [0,2pi] (range = 2)

    Parameters:
                  - x vector or matrix in radian to convert
                  - range: 1 [-pi,pi]
                           2 [0, 2pi]

    Returns:
                  - return and element like x of converted values

    """
    x = x % (2 * pi)

    if rangeVal == 1:
        underPi = x > pi
        x[underPi] = x[underPi] - (2 * pi)

    return x


########################################################################
## Descriptive statistics
########################################################################


def circ_mean(alpha, weight=None, dim=0):
    """
    function that calculate the circular mean of a given vector
    """
    alpha = np.asarray(alpha)
    if weight is None:
        weight = np.ones_like(alpha)

    # compute weighted sum of cos and sin of angles
    r = bn.nansum(np.multiply(weight, np.exp(1j * alpha)), dim)
    mu = np.angle(r)

    return mu


def circ_r(alpha, weight=None, d=False, dim=0):
    """"""
    alpha = np.asarray(alpha)
    if weight is None:
        weight = np.ones_like(alpha)

    # compute weighted sum of cos and sin of angles
    r = bn.nansum(np.multiply(weight, np.exp(1j * alpha)), dim)

    r = abs(r) / bn.nansum(weight, dim)

    # from matlab circ_r function
    # for data with known spacing, apply correction factor
    # to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)

    if d:
        c = d / 2.0 / np.sin(d / 2)
        r = c * r

    return r


def circ_std(alpha, weight=None, d=False, dim=0):
    """
    Compute circular Standard Deviation
    """
    alpha = np.asarray(alpha)
    if weight is None:
        weight = np.ones_like(alpha)

    # compute the resultant vector of the data
    r = circ_r(alpha, weight, d, dim)

    # calculate angular deviation
    s = np.sqrt(2 * (1 - r))

    # calculate circular standard deviation
    s0 = np.sqrt(-2 * np.log(r))

    return s, s0


########################################################################
## Statistics
########################################################################


def corr_cc(alpha, beta, tail="two-sided", uniformity_correction=False):
    """
    Function that  compute correlation between two circular variables
    Inputs:
            - alpha (1D array):
                    first vector of circular variables (in radian)
            - beta (1D array):
                    second vector of circular variables (in radian)
            - tail (string, default: 'two-sided'):
                    determine is on or two sided p-value should be returned
            - uniformity_correction (bool, default=False):
                    Define if a correction for uniform variables
                    should be used.
    Return:
            - rho (float): circular correlation coefficient
            - pval(float): two sided p-value

    Adapted from Ref[1-3]

    From Ref [3] Jammalamadaka, S. R., & Sengupta, A. (2001).
    Topics in circular statistics p. 180

    from hippocampy.stats import circ_stats
    import numpy as np

    Theta = [356,97,211,232,343,292,157,302,335,302,324,85,324,340,157,238,254,146,232,122,329]
    Phi = [119,162,221,259,270,29,97,292,40,313,94,45,47,108,221,270,119,248,270,45,23]

    rTheta = np.deg2rad(Theta)
    rPhi = np.deg2rad(Phi)

    r, pval = circ_stats.corr_cc(rTheta,rPhi)
    print(round(r, 3), round(pval, 4))
    0.27 0.2247

    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    assert alpha.size == beta.size, "alpha and beta should have the same length"

    # initialise variables
    n = alpha.size

    sin_alpha_z = np.sin(alpha - circ_mean(alpha))
    sin_beta_z = np.sin(beta - circ_mean(beta))

    sin_alpha_z_2 = sin_alpha_z ** 2
    sin_beta_z_2 = sin_beta_z ** 2

    if not uniformity_correction:
        # compute correlation coefficient from p 176 of Ref in description
        num = bn.nansum(sin_alpha_z * sin_beta_z)
        denom = np.sqrt(bn.nansum(sin_alpha_z_2) * bn.nansum(sin_beta_z_2))
        rho = num / denom
    else:
        # in case of uniformity of a distribution the formula described p177
        # of Ref[3] as in Ref[2]
        R_plus = np.abs(bn.nansum(np.exp((alpha + beta) * 1j)))
        R_minus = np.abs(bn.nansum(np.exp((alpha - beta) * 1j)))
        denom = 2 * np.sqrt(bn.nansum(sin_alpha_z_2) * bn.nansum(sin_beta_z_2))
        rho = (R_minus - R_plus) / denom

    # compute pvalue
    l20 = bn.nanmean(sin_alpha_z_2)
    l02 = bn.nanmean(sin_beta_z_2)
    l22 = bn.nanmean(sin_alpha_z_2 * sin_beta_z_2)

    tstat = np.sqrt((n * l20 * l02) / l22) * rho

    pval = 2 * (1 - norm.cdf(abs(tstat)))
    pval = pval / 2 if tail == "one-sided" else pval

    return rho, pval


def corr_cl(x, theta, tail="two-sided"):
    """
    circular linear correlation between a linear variable x and circular theta
    """
    from scipy.stats import pearsonr, chi2

    x = np.asarray(x)
    theta = np.asarray(theta)

    assert x.size == theta.size, "x and theta should have the same length"
    assert tail in ["one-sided", "two-sided"], "Tail should be one-sided or two-sided "

    n = x.size

    # Compute pearson correlation forsin and cos of angular data
    rxs = pearsonr(theta, np.sin(x))[0]
    rxc = pearsonr(theta, np.cos(x))[0]
    rcs = pearsonr(np.sin(x), np.cos(x))[0]

    r = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))

    # Calculate p-value
    pval = chi2.sf(n * r ** 2, 2)
    pval = pval / 2 if tail == "one-sided" else pval
    return r, pval


def lin_circ_regress(x, phi, bound=None):
    """
    Linear-circular regression as described in Kempter et al 2012

    Parameters:
                - x: linear variable
                - phi: circular variable
                - bound: bounds to contrains the regression search (eg: if we
                    know in advance our slope should be negative,...)

    Returns:
                - rho: Kempter linear-Circular R value
                - slope: slope best fit
                - phi0: phase offset (value at x=0)
                - p_c: p value calculated numerically from Kempter et al 2012 p122

    Example:
    x = np.asarray([107, 46, 33, 67, 122, 69, 43, 30, 12, 25, 37,
        69, 5, 83, 68, 38, 21, 1, 71, 60, 71, 71, 57, 53, 38, 70, 7, 48, 7, 21, 27])
    phi = np.asarray([67, 66, 74, 61, 58, 60, 100, 89, 171, 166, 98,
        60, 197, 98, 86, 123, 165, 133, 101, 105, 71, 84, 75, 98, 83, 71, 74, 91, 38, 200, 56])
    phi = np.deg2rad(phi)

    print(lin_circ_regress( x, phi))
    (-0.49, -0.012, 2.28, 0.018)

    Acknowledgments:
    circ_regress_cb by A. Jeewajee and C. Barry
    CircularRegression by M. Zugaro

    """
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # verify inputs
    assert x.size == phi.size, "f and g should have the same size"
    assert isradians(phi) != 0, "circular variable should be in radian"

    # estimate the slope and constrain it to a particular interval
    if bound is None:
        max_slope = (4 * np.pi) / (np.max(x) - np.min(x))
        bound = [-max_slope, max_slope]
    else:
        assert bound.size == 2, "bound should be in the format [low_bound, high_bound]"

    lambda_resultant_length = lambda a: _resultant_length(a, x, phi)
    slope = fminbound(lambda_resultant_length, bound[0], bound[1], disp=False)

    # estimate the intercept with Kempter 2012 Eq 2
    C = bn.nansum(np.cos(phi - slope * x))
    S = bn.nansum(np.sin(phi - slope * x))
    phi0 = np.arctan2(S, C)

    # and compute the circular linear correlation with Kempter 2012 Eq 3
    # first circularise the linear variable
    theta = np.mod(np.abs(slope) * x, 2 * np.pi)

    # compute the circular mean of each variables
    phi_bar = circ_mean(phi)
    theta_bar = circ_mean(theta)

    # Compute eq 3
    Num = bn.nansum(np.sin(phi - phi_bar) * np.sin(theta - theta_bar))
    Denum = np.sqrt(
        bn.nansum(np.sin(phi - phi_bar) ** 2)
        * bn.nansum(np.sin(theta - theta_bar) ** 2)
    )
    rho = Num / Denum

    # Now calculate p value according to Kempter equation p122
    n = phi.size
    lambda02 = 1 / n * bn.nansum(np.sin(phi - phi_bar) ** 2)
    lambda20 = 1 / n * bn.nansum(np.sin(theta - theta_bar) ** 2)
    lambda22 = (
        1 / n * bn.nansum(np.sin(phi - phi_bar) ** 2 * np.sin(theta - theta_bar) ** 2)
    )

    z = rho * np.sqrt((n * lambda02 * lambda20) / lambda22)
    p_c = 1 - erf(np.abs(z) / np.sqrt(2))

    return rho, slope, phi0, p_c


def _resultant_length(a, x, phi):
    """
    helper function for lin_circ_regress
    """
    n = x.size
    G = (1 / n) * bn.nansum(np.cos(phi - a * x))
    D = (1 / n) * bn.nansum(np.sin(phi - a * x))
    # we will return -R as will will then search to minimise the function
    # minimizing -R == maximinsing R
    return -np.sqrt(G ** 2 + D ** 2)


########################################################################
## other
########################################################################


def cemd(f, g, period=[0, 2 * pi]):
    """
    Calculate circular earth mover distance between two histograms
    of circular distributions

    Input:
            - f and g two histograms of circular distributions
            - period: is a variable defining on which domain we calculate the distance
            (eg: [0, 2*pi] for radian, [0, 24] for hours, ....)
    Output:
            - CEMD: circular Wassertein distance

    References:
    Rabin et al 2011
    """
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)

    assert f.size == g.size, "f and g should have the same length"

    # initialise variables
    n = len(f)
    D = np.empty_like(f)
    idx = np.arange(n)

    # normalize histograms just in case it was not done before
    # histograms should be probability distributions such as sum(f) = sum(g) = 1
    f /= bn.nansum(f)
    g /= bn.nansum(g)

    # loop on al the possible starting positions
    for k in idx:
        # shift indexes
        idx_k = np.roll(idx, k - 1)

        # calculate cumulative histograms
        F_k = np.cumsum(f[idx_k])
        G_k = np.cumsum(g[idx_k])

        F_k /= bn.nansum(F_k)
        G_k /= bn.nansum(G_k)

        # calculate sum of distances
        D[k] = bn.nansum(abs(F_k - G_k))

    # divide by n and multiply by "period" in order to set the max distance
    # to this value.
    D /= n * np.diff(period)

    return bn.nanmin(D)
