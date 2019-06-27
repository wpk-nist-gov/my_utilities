"""
labeled sums along axis
"""
from __future__ import absolute_import
from builtins import *

import numpy as np
from numba import jit

#--------------------------------------------------
# this uses explicit types.  Not much faster than python object code, so not worth the trouble
# import itertools
# def _get_sig_labeled_sum(x_dim='[:,::1]', labels_dim='[::1]', out_dim='[:,::1]'):
#     c_list = ['complex64','complex128']
#     f_list = ['float32', 'float64']
#     i_list = ['int32','int64']

#     L_f = list(itertools.product(f_list, i_list, f_list))
#     L_c = list(itertools.product(c_list, i_list, c_list))
#     L_i = list(itertools.product(i_list, i_list, i_list))
#     L = L_f + L_c + L_i
#     f = lambda x: 'void({x}{x_dim}, {labels}{labels_dim}, {out}{out_dim})'.format(x=x[0],labels=x[1],out=x[2], 
#                                                                                  x_dim=x_dim, labels_dim=labels_dim,
#                                                                                  out_dim=out_dim)
#     return map(f, L)

# @jit(_get_sig_labeled_sum(), nopython=True)
# def _labeled_sum_out(x, labels, out):
#     for i in range(x.shape[0]):
#         label = labels[i]
#         for j in range(x.shape[1]):
#             out[label, j] += x[i,j]


@jit
def _labeled_sum_out(x, labels, out):
    """
    find the labeled sum for the 2d array x
    """
    for i in range(x.shape[0]):
        label = labels[i]
        for j in range(x.shape[1]):
            out[label, j] += x[i, j]


def _broadcast_labeled_sum(x, labels, nlabels, assure_order=True, out=None):
    """
    calculate labeled sum along first axis

    Parameters
    ----------
    x : ndarray
        array to sum.
        shape=(n, nd0, nd1, ...)

    labels : 1d array
        array of labels.
        length = n

    nlabels : int
        number of unique labels

    assure_order : bool (Default True)
        if True, check that `out` is c-contiguous,
        and force `x` to c-contiguous

    out : ndarray (optional)
        if present, use this array for output
        must be of shape (nlabels, nd0, nd1, ...)

    Returns
    -------
    output : ndarray
        shape=(nlabels, nd0, nd1,...)
        output[label,...] = x[labels==label, ...].sum(axis=0)
        only returned if out is not supplied
    """

    shape_out = (nlabels, ) + x.shape[1:]
    if out is not None:
        assert out.shape == shape_out
        out.fill(0)
        output = out
    else:
        output = np.zeros(shape_out, dtype=x.dtype)

    if assure_order:
        assert output.flags['C_CONTIGUOUS']
        x = np.ascontiguousarray(x)

    # reshape x
    if x.ndim == 2:
        xr = x
    else:
        xr = x.reshape(x.shape[0], -1)

        # check labels shape
    assert (xr.shape[:1] == labels.shape)

    # reshape output
    outr = output.reshape(nlabels, -1)
    # calculate
    _labeled_sum_out(xr, labels, outr)

    if out is None:
        return output


def broadcast_labeled_sum(x,
                          labels,
                          nlabels,
                          assure_order=True,
                          axis=0,
                          roll=None,
                          out=None):
    """
    calculate labeled sum along axis

    Parameters
    ----------
    x : ndarray
        array to sum

    labels : 1d array
        array of labels

    nlabels : int
        number of unique labels

    assure_order : bool (Default True)
        parameter to `broadcast_labeld_sum`

    axis : int (Default 0)
        axis to sum along with labels

    out : ndarray
        store results here. If supplied, cannot perform `roll` below

    roll : int (Default None)
        By default, return is of shape (nlabels, nd0, nd1,..), where `ndi` are 
        axis sizes for all but `axis` dimensions. if roll is an integer, return
        np.rollaxis(output, 0, roll)

    Returns
    -------
    output : ndarray

    """

    if out is not None and roll is not None:
        raise ValueError('cannot supply `out` and `roll`')

    if axis != 0:
        x = np.rollaxis(x, axis, 0)

    output = _broadcast_labeled_sum(
        x, labels, nlabels,
        assure_order=assure_order,
        out=out)

    if roll is not None:
        output = np.rollaxis(output, 0, roll)

    if out is None:
        return output


def broadcast_labeled_sum_bincount(
        x, labels,
        axis=0, nlabels=None,
        roll=False,
        label_matrix=False):
    """
    sum along axis with specified labels

    Parameters
    ----------
    x : ndarray
        input array to sum

    labels : array
        labels (or bins) for summing. if labels is a 1d array, then broadcast 
        labels across axes other than `axis`.  Else, labels is same shape as labels

    axis : int (default 0)

    roll : bool or int (default False)
        if True, roll binned axis to `axis`.
        if False, leave label axis at the end
        if integer, roll label axis to this positon (see numpy.rollaxis)

    label_matrix : bool (default False)
        if True, return label matrix

    Returns
    -------
    output : ndarray
        if roll, then output.shape[axis] = nlabels, and other dimensions are same as x
        else, output.shape = (...x.shape[axis-1], x.shape[axis+1],...,nlabels)

    label_m : ndarray
        matrix of labels (broadcast and shifted from 1d labels)
        if `label_matrix` is True in call.
    """

    x = np.asarray(x)
    labels = np.asarray(labels)

    # shapes
    shape_other = list(x.shape)

    # everthing less axis
    n = shape_other.pop(axis)
    shape_other = tuple(shape_other)
    # length of everything else
    n_other = np.prod(shape_other)

    if nlabels is None:
        nlabels = labels.max()

        # label matrix
    if labels.ndim == 1:
        # create label matrix
        assert len(labels) == x.shape[axis]

        broadcast_index = tuple(None if i != axis else slice(None)
                                for i in range(x.ndim))
        offset_shape = tuple(x.shape[i] if i != axis else -1
                             for i in range(x.ndim))

        labels_m = (np.broadcast_to(labels[broadcast_index], x.shape
                                    )  # broadcast across meta data
                    + np.arange(n_other).reshape(offset_shape) *
                    nlabels  # shift each meta column
                    )
    else:
        # use labels as label matrix
        assert labels.shape == x.shape
        labels_m = labels

    if nlabels is None:
        minlength = labels_m.max()
    else:
        minlength = nlabels * n_other

        # binned sum
    shape_output = tuple(list(shape_other) + [nlabels])
    output = (np.bincount(
        labels_m.ravel('C'),
        x.ravel('C'), minlength=minlength)
              .reshape(shape_output, order='C'))

    # roll
    if type(roll) is bool:
        if roll:
            output = np.rollaxis(output, -1, axis)
    else:
        output = np.rollaxis(output, -1, roll)

#    if roll:
#        output = np.rollaxis(output, -1, axis)

    if label_matrix:
        return output, labels_m
    else:
        return output

