from scipy.optimize import curve_fit
import bottleneck as bn
import numpy as np


def lognormal(x: np.array, B: float, mu: float, sigma: float):
    """Generate lognormal distribution"""
    return (
        B
        * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
        / (x * sigma * np.sqrt(2 * np.pi))
    )


def logifunc(x: np.array, A: float, x0: float, k: float, off: float):
    return A / (1 + np.exp(-k * (x - x0))) + off


def fit_pdf(
    dist, dist_type, *, n_bins: int = 100, n_bins_out: int = None, max_dist=None
):

    if dist_type not in ["logistic", "lognormal"]:
        raise ValueError(f"Distribution type {dist_type} not recognized")

    if max_dist is None:
        max_dist = bn.nanmax(dist)

    if n_bins_out is None:
        n_bins_out = n_bins

    if dist_type == "logistic":
        dist_to_fit = logifunc
    elif dist_type == "lognormal":
        dist_to_fit = lognormal

    edges = np.linspace(0, max_dist + 1e-5, n_bins)
    centers = np.linspace(0.001, max_dist, n_bins - 1)
    centers_out = np.linspace(0.001, max_dist, n_bins_out)

    pdf, _ = np.histogram(dist, edges, density=True)
    pdf = pdf + np.finfo(pdf.dtype).eps

    sol, _ = curve_fit(dist_to_fit, centers, pdf)

    # calculate errors of the model
    E = bn.nanmean((dist_to_fit(centers, *sol) - pdf) ** 2)

    return dist_to_fit(centers_out, *sol), centers_out, sol, E

