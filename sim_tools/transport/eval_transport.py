#!/usr/bin/env python

#from __future__ import with_statement

from optparse import OptionParser
import sys
# import os
# import commands
import netCDF4 as nc
import numpy as np
import sys
import xarray as xr


def gen_mask_trans(t, y, t_min, n_skip=0, t_max=None):
    """
    mask MSD
    """

    _filter = (t >= t_min)
    if t_max is not None:
        _filter = (_filter) & (t <= t_max)

    yf = y[n_skip:, _filter]

    return yf


def gen_eval_trans_WA(yf):
    """
    calculate transport from MSD/t
    
    assumes yf is properly setup
    MSD/t
    masked tmin/max"""

    nrec = yf.shape[0]

    if nrec == 1:
        return np.average(yf), np.std(yf), nrec
    
    ave = np.average(yf, axis=0)
    w = np.var(yf, axis=0, ddof=0)


    # rescaled to maximum
    w = w / w.max()

    if np.any(w < 0.0):
        raise ValueError('bad weight')

    w = 1. / w





    Sum_V = np.sum(ave * w)
    Sum_InvSigSq = np.sum(w)
    Sum_VSq_o_SigSq = np.sum(w * ave * ave)

    Ave = Sum_V / Sum_InvSigSq
    Var = 1. / (nrec - 1) * (Sum_VSq_o_SigSq / Sum_InvSigSq - Ave**2)



    return (Ave, np.sqrt(Var), nrec)


def eval_transport_WA(t, y, t_min, n_skip=0, t_max=None):
    """ 
    calculate the transport property from MSD/t
    
    t=time array
    y=MSD/t array (shape=(nrec,ntime))
    t_min= minimum time"""

    if n_skip >= y.shape[0]:
        return np.nan, np.nan, np.nan

    yf = gen_mask_trans(t, y, t_min, n_skip, t_max)

    return gen_eval_trans_WA(yf)


def eval_transport_AveTmin(t, y, t_min, n_skip=0, t_max=None):
    """ 
    calculate the transport property from MSD/t
    
    t=time array
    y=MSD/t array (shape=(nrec,ntime))
    t_min= minimum time"""

    if n_skip >= y.shape[0]:
        return np.nan, np.nan, np.nan

    yf = gen_mask_trans(t, y, t_min, n_skip, t_max)

    ave = np.average(yf, axis=1)
    nrec = yf.shape[0]

    return np.average(ave), np.std(ave, ddof=1), nrec


def f_get_t_y(fname, t_name, y_name, y_slice, div_by_t):
    root = nc.Dataset(fname)

    t = root.variables[t_name][:]
    y = root.variables[y_name][y_slice]

    if div_by_t:
        y = y / t

    return t, y



def eval_transport_WA_ds(msd, t_min, n_skip=0, t_max=None, rec_dim='rec', t_dim='time', fmt='dataset',  **kwargs ):
    """
    calculate transport from xarray dataset


    Parameters
    ----------
    msd : xarray
        dataset containing MSD/t data

    t_min : float
        minimum time

    n_skip : int
        number of records to skip

    t_max : float
        maximum time to consider

    rec_dim : str
        name of record dimension

    t_dim : str
        name of time dimension

    fmt : str
        one of 'array', 'dataset'
        if array, output ave, std, nrec
        if dataset, output dataset


    Returns
    -------
    ave, std, nrec : if fmt=='array'

    output : xarray (if fmt=='dataset')
        transport
    """

    if len(msd[rec_dim]) <= n_skip:
        raise ValueError('not enough records')


    # mask data
    time_mask = msd[t_dim] >= t_min
    ds = msd.isel(**{rec_dim : slice(n_skip,None)}).sel(**{t_dim : time_mask})

    if t_max is not None:
        ds = ds.sel(**{t_dim : ds[t_dim] < t_max})

    # calculate transport
    nrec = len(ds[rec_dim])
    if nrec == 0:
        Ave = np.nan
        Std = np.nan

    elif nrec == 1:
        Ave = ds.mean([rec_dim, t_dim])
        Std = ds.std([rec_dim, t_dim])

    else:
        ave = ds.mean(rec_dim)
        w = ds.var(rec_dim, ddof=0)

        # rescale by max
        w = w / w.max().values

        if np.any( w < 0.0):
            raise ValueError('bad weight')

        w = 1. / w
        S1 = (ave * w).sum(t_dim)
        V1 = w.sum(t_dim)
        V2 = (w * w).sum(t_dim)

        Ave = S1 / V1
        # weighted variance from https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
        Std = np.sqrt((w * (ave - Ave)**2).sum(t_dim) / (V1 - V2/V1))


    # output
    if fmt.lower() == 'array':
        return Ave, np.sqrt(Var), nrec
    elif fmt.lower() == 'dataset':
        output = xr.Dataset({'ave' : Ave,
                             'std': Std,
                             'tmin': t_min,
                             'nrec': nrec,
                             'ntime': len(ds[t_dim])}).set_coords('tmin')
        output['ave'].attrs['long_name'] = 'transport property calculated by weighted average method'
        output['std'].attrs['long_name'] = 'weighted standard deviation across time'
        return output
    else:
        raise ValueError('unknown fmt {}'.format(fmt))













eval_dic = {'wa': eval_transport_WA, 'a': eval_transport_AveTmin}

def f_get_options(args_in):

    usage = "usage: %prog [options]"

    parser = OptionParser(usage=usage)
    parser.add_option('-m',
                      '--t_min',
                      dest='t_min',
                      action='store',
                      default=None,
                      help='minimum time',
                      type='float')

    parser.add_option('-M',
                      '--t_max',
                      dest='t_max',
                      action='store',
                      default=None,
                      type='float',
                      help='maximum time [default %default]')

    parser.add_option('-s',
                      '--n_skip',
                      dest='n_skip',
                      action='store',
                      default=0,
                      type='int',
                      help='number of records to skip [default %default]')

    parser.add_option('-i',
                      '--inner_index',
                      dest='i_index',
                      action='store',
                      default=None,
                      type='string',
                      help="""
                      comma separated list for "inner" index
                      """)

    parser.add_option('-t',
                      '--t_name',
                      dest='t_name',
                      action='store',
                      default='time',
                      help='name of time array [default=%default]')

    parser.add_option('-y',
                      '--y_name',
                      dest='y_name',
                      action='store',
                      default='MSD',
                      help='name of y array [default=%default]')

    parser.add_option('--div_by_t',
                      dest='div_by_t',
                      action='store_true',
                      default=False,
                      help='divide by time_array [default=%default]')

    parser.add_option('-I',
                      '--integrate',
                      dest='integrate',
                      action='store_true',
                      default=False,
                      help='integrate ACF (and divide by t)')

    parser.add_option('-e',
                      '--eval_type',
                      dest='eval_type',
                      default='wa',
                      help="""
                      evaluation type 
                      [wa=Weighted average, a=average] 
                      [default=%default]
                      """)

    (opt, args) = parser.parse_args(args_in)

    error = False

    if opt.t_min is None:
        print("require t_min")
        error = True

    if len(args) <= 0:
        print( "need file(s)")
        error = True

    y = [slice(None, None)]
    if opt.i_index is None:
        a = y * 2
    else:
        a = [int(x) for x in opt.i_index.split(',')]
        a = y + a + y

    opt.y_slice = a

    if error:
        sys.exit(1)

    return opt, args


if __name__ == '__main__':

    if len(sys.argv) < 2:
        f_get_options(['-h'])
    else:

        (opt, args) = f_get_options(sys.argv[1:])

        for file_name in args:

            print( file_name)

            t, y = f_get_t_y(file_name, opt.t_name, opt.y_name, opt.y_slice,
                             opt.div_by_t)

            val = eval_dic[opt.eval_type](t, y, opt.t_min, opt.n_skip,
                                          opt.t_max)
            print( '%.8e %.8e %i' % (val))
