"""
some utilities to work with xarray objects
"""

import numpy as np
import xarray as xr


def strip_coords(X, coords=None, inplace=False, as_str=True):
    """
    strip blanks from string coordinates

    Parameters
    ----------
    X : DataArray or Dataset

    coords : iterable (Default None)
        Iterable of coordinates to alter.
        If `None`, apply to all (string) coordinates

    inplace : bool (Default False):
        if True, do inplace modification

    as_str : bool, default=True
        if `True`, apply .astype(str) to output.
        This helps if input has b'foo' types (which are annoying to work with)


    """
    if inplace:
        out = X
    else:
        out = X.copy()

    if coords is None:
        coords = out.coords.keys()

    for k in coords:
        if out[k].dtype.kind == 'S':
            o = np.char.strip(out[k])
            if as_str:
                o = o.astype(str)
            out[k] = o

    if not inplace:
        return out


def where(self, condition, *args, **kwargs):
    """
    perform inplace where

    Parameters
    ----------
    self : dataset or datarray
        must have `where`` method
    condition: mask or function
        condition to apply
    *args, **kwargs: arguments to self.where

    Returns
    -------
    output : self.where(condition, *args, **kwargs)


    Usage
    -----
    self.pipe(where, lambda x: x > 0.0)
    """

    if not hasattr(self, 'where'):
        raise AttributeError('self must have `where` method')

    if callable(condition):
        return self.where(condition(self), *args, **kwargs)
    else:
        return self.where(condition, *args, **kwargs)


def average(x, w=None,
            dim=None, axis=None,
            var=False, unbiased=True, std=False,
            name=None,
            mask_null=True):
    """
    (weighted) average of DataArray

    Parameters
    ----------
    x : xarray.DataArray
        array to average over
    w : xarray.DataArray, optional
        array of weights
    dim : str or list of strings, optional
        dimensions to average over.  See `xarray.DataArray.sum`
    axis : int or list of ints, optional
        axis to average over. See `xarray.DataArray.sum`
    var : bool, default=False
        If `True`, calculate weighted variance as well
    std : bool, default=False
        If `True`, return standard deviation, i.e., `sqrt(var)`
    unbiased : bool, default=True
        If `True`, return unbiased variance
    name : str, optional
        if supplied, name of output average. Variance is named 'name_var' or 'name_std'
    mask_null : bool, default=True
        if `True`, mask values where x and w are all null across `dim` or `axis`.
        This prevents zero results from nan sums.

    Returns
    -------
    average : xarray.DataArray
        averaged data
    err : xarray.DataArray, optional
        weighted variance if `var==True` or standard deviation if `std==True`.
    """
    assert type(x) is xr.DataArray
    if w is None:
        w = xr.ones_like(x)
    assert type(w) is xr.DataArray
    # only consider weights with finite x
    # note that this will reshape w to same shape as x as well
    w = w.where(np.isfinite(x))
    # scale w
    w = w / w.sum(dim=dim, axis=axis)

    # output names
    if name:
        var_name = name + ('_std' if std else '_var')
    else:
        var_name = None

    # mean
    m1 = (w * x).sum(dim=dim, axis=axis)

    if mask_null:
        msk = (~x.isnull().all(dim=dim, axis=axis)) & (~w.isnull().all(dim=dim, axis=axis))
        m1 = m1.where(msk)
    

    # variance
    if var or std:
        m2 = (w * (x - m1)**2).sum(dim=dim, axis=axis)
        if unbiased:
            w1 = 1.0
            w2 = (w * w).sum(dim=dim, axis=axis)
            m2 *= w1 * w1 / (w1 * w1 - w2)

        if std:
            m2 = np.sqrt(m2)

        if mask_null:
            m2 = m2.where(msk)

        return m1.rename(name), m2.rename(var_name)

    else:
        return m1.rename(name)
