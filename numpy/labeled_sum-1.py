"""
labeled sums along axis
"""
from __future__ import absolute_import
from builtins import *

import numpy as np
from numba import jit


@jit(nopython=True)
def _labeled_sum(x, labels, nlabels):
    out = np.zeros((nlabels, x.shape[1]), dtype=x.dtype)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[labels[i], j] += x[i, j]
    return out


def _broadcast_labeled_sum(x, labels, nlabels):
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

    Returns
    -------
    output : ndarray
        shape=(nlabels, nd0, nd1,...)

    """
    # only works for 2d arrays with labels along axis==0
    if x.ndim == 2:
        xr = x
    else:
        xr = x.reshape(x.shape[0], -1)

    assert (xr.shape[:1] == labels.shape)
    # get labeled sums
    output = _labeled_sum(xr, labels, nlabels)

    #return reshaped output
    return output.reshape(nlabels, *x.shape[1:])


def broadcast_labeled_sum(x, labels, nlabels, axis=0, roll=None):
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

    axis : int (Default 0)
        axis to sum along with labels

    roll : int (Default None)
        By default, return is of shape (nlabels, nd0, nd1,..), where `ndi` are 
        axis sizes for all but `axis` dimensions. if roll is an integer, return
        np.rollaxis(output, 0, roll)

    Returns
    -------
    output : ndarray
        labeled sum
    """
    if axis != 0:
        x = np.rollaxis(x, axis, 0)

    output = _broadcast_labeled_sum(x, labels, nlabels)

    if roll is not None:
        output = np.rollaxis(output, 0, roll)
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
