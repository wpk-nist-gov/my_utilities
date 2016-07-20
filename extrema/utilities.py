"""
Collection of utilites to find extremas
"""

import numpy as np
from scipy.interpolate import UnivariateSpline as UVSpline

def find_extrema_spline(x,y,k=4,s=0,**kwargs):
    """
    find local extrema of y(x) by taking derivative of spline

    Parameters
    ----------

    x,y : array-like
      find extrema of y(x)

    k : int
      order of spline interpolation [must be >3]

    s : number
      s parameter sent to scipy spline interpolation (used for smoothing)

    **kwargs : extra arguments to UnivariateSpline


    Returns
    -------
    sp : UnivariateSpline object

    x_max,y_max : array
      value of x and y at extrema(s)
    """


    sp = UVSpline(x,y,k=k,s=s,**kwargs)

    x_max = sp.derivative().roots()

    y_max = sp(x_max)

    return sp,x_max,y_max


def classify_extrema(xloc,sp,eps=1e-8):
    """
    classify extrema found from spline
    
    Parameters
    ----------
    xloc : scalar or array-like 
     x location of extrema to evalute

    sp : UnivariateSpline

    eps : float 
     perterbation to consider

    Returns
    -------
    ismax : bool array
      True if xloc is a maxima, False if minima
    
    """

    x = np.atleast_1d(xloc)

    #forward difference
    return  sp(x+eps) < sp(x)

    

from scipy.signal import argrelextrema,argrelmax,argrelmin
def find_maxima_data(y,**kwargs):
    """
    find local maxima

    Parameters
    ----------
    y : array 
      values to consider

    **kwargs : arguments to scipy.signal.argrelmax
     [axis,order,mode]


    Returns
    -------
    extrema : tuple of ndarrays
     Indices of the maxima in arrays of integers.  ``extrema[k]`` is
     the array of indices of axis `k` of `data`.  Note that the
     return value is a tuple even when `data` is one-dimensional.    
    """

    return argrelmax(y,**kwargs)

def find_minima_data(y,**kwargs):
    """
    find local minima

    Parameters
    ----------
    y : array 
      values to consider

    **kwargs : arguments to scipy.signal.argrelmin
     [axis,order,mode]


    Returns
    -------
    extrema : tuple of ndarrays
     Indices of the minima in arrays of integers.  ``extrema[k]`` is
     the array of indices of axis `k` of `data`.  Note that the
     return value is a tuple even when `data` is one-dimensional.    
    """

    return argrelmin(y,**kwargs)


    



    
    
     
      

