from typing import Tuple
import bottleneck as bn
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from hippocampy.utils.nan import remove_nan


def lognormal(x: np.array, B: float, mu: float, sigma: float):
    """Generate lognormal distribution"""
    return (
        B
        * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
        / (x * sigma * np.sqrt(2 * np.pi))
    )


def logifunc(x: np.array, A: float, x0: float, k: float, off: float):
    return A / (1 + np.exp(-k * (x - x0))) + off


def linfunc(x: np.array, a: float, x0: float):
    return (a * x) + x0


def fit_pdf(
    dist, dist_type, *, n_bins: int = 100, n_bins_out: int = None, max_dist=None
):

    if dist_type not in ["linear", "logistic", "lognormal"]:
        raise ValueError(f"Distribution type {dist_type} not recognized")

    if max_dist is None:
        max_dist = bn.nanmax(dist)

    if n_bins_out is None:
        n_bins_out = n_bins

    if dist_type == "logistic":
        dist_to_fit = logifunc
    elif dist_type == "lognormal":
        dist_to_fit = lognormal
    elif dist_type == "linear":
        dist_to_fit = linfunc

    edges = np.linspace(0, max_dist + 1e-5, n_bins)
    centers = np.linspace(0.001, max_dist, n_bins - 1)
    centers_out = np.linspace(0.001, max_dist, n_bins_out)

    pdf, _ = np.histogram(dist, edges, density=True)
    pdf = pdf + np.finfo(pdf.dtype).eps

    sol, _ = curve_fit(dist_to_fit, centers, pdf)

    # calculate errors of the model
    E = bn.nanmean((dist_to_fit(centers, *sol) - pdf) ** 2)

    return dist_to_fit(centers_out, *sol), centers_out, sol, E


def fit(
    x: np.ndarray, y: np.ndarray, dist_type: str = "logistic", n_bins_out: int = 100
):
    if dist_type not in ["logistic", "lognormal"]:
        raise ValueError(f"Distribution type {dist_type} not recognized")
    if n_bins_out is None:
        n_bins_out = n_bins


def robust_regression(
    x_reg: np.ndarray, y_reg: np.ndarray, n_bins: int = 100
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    convenience wrapper to perform robust regression

    Parameters
    ----------
    x_reg : np.ndarray
        x_value to be regressed
    y_reg : np.ndarray
        y_value to be regressed
    n_bins : int, optional
        number of bins to use to output predicted values, by default 100

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        _description_
    """

    x_reg, y_reg = remove_nan(x_reg, y_reg, paired=True)

    ransac = RANSACRegressor()
    ransac.fit(x_reg.reshape(-1, 1), y_reg.reshape(-1, 1))

    x_pred = np.linspace(bn.nanmin(x_reg), bn.nanmax(x_reg), n_bins)[:, None]
    y_pred = ransac.predict(x_pred)

    return ransac.estimator_.coef_, x_pred, y_pred

