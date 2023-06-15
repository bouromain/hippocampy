import matplotlib.pyplot as plt
import bottleneck as bn
import numpy as np
from hippocampy.assemblies import calc_template


def joy_plot(F: np.ndarray, T=None, n_traces=20):

    plt.figure


def line_error(M:np.ndarray, x:np.ndarray = None, color:str=None,alpha:float= 0.4, var_type:str = 'ste'):
    """
    line_error return a axis handle of a line plot with shaded error overlaid 
    around this average/median curve

    Parameters
    ----------
    M : np.ndarray
        input matrix [n_sample, n_trial]
    x : np.ndarray, optional
        x vector [n_samples], by default None
    color : str, optional
        force a specific color to the plot, by default None
    alpha : float, optional
        alpha value for the error, by default 0.4
    var_type : str, optional
        type of error to plot ['ste', 'std'], by default 'ste'

    Returns
    -------
    ax
        figure axis
        

    Raises
    ------
    ValueError
        if x and M size are inconsistent
    ValueError
        _description_
    """

    if var_type not in ['ste', 'std']:
        raise ValueError('var_type should either be ste: standard error or std: standard deviation')
    m = bn.nanmean(M,axis=0)
    ste = bn.nanstd(M,axis=0,ddof=1) / np.sqrt(M.shape[0])
    
    if x is  None:
        x = np.arange(len(m))
    else:
        if len(m) != len(x):
            raise ValueError('Size mismatch between x and input matrix M')

    plt.plot(x, m,color=color)
    plt.fill_between(x,m-ste, m+ste,alpha=alpha, antialiased=True, edgecolor=color, facecolor=color)
    return plt.gca()

