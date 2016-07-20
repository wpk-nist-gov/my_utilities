"""
some useful statistics things
"""

import scipy.stats
import numpy as np


def StandardError(x, conf=0.95, ddof=1):
    """
    calculate the standard error over array

    Parameters
    ----------
    x : array
     array of values to consider

    conf : float
     confidance interval to apply [default=0.95]

    ddof : int
     degree of freedom correction.  dof=size(x)-ddof

    Returns
    -------
    SE : float
     standard error.  Calculated with students t distribution
    """
    # number of non nan values
    n = np.count_nonzero(~np.isnan(x))
    if (n <= 1):
        SE = 0
    else:
        #ddof=0, standard dev, ddof=1, sample standard dev
        SE = np.nanstd(x, ddof=1) / np.sqrt(n) \
             * scipy.stats.t.isf(0.5 * (1 - conf), n - ddof)
    return SE


StandardError.__name__ = 'SE'
