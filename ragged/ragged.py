"""
routines to pack/unpack ragged arrays
"""

from itertools import izip
import numpy as np


class ragarray(object):
    """
    ragged array class. create a container for 2d ragged arrays

    Attributes
    ----------
    x: array like
     data

    starts: array like
     start of each data segment

    lens: array like
     length of each data segment


    Methods
    -------
    """

    def __init__(self,x,starts=None,lens=None):

        self._data = None
        self._starts = None
        self._lens = None

        if type(x) is list:
            #pack data
            self._data, self._starts, self._lens = pack_list(x)
        else:
            if starts is None or lens is None:
                raise ValueError('need starts and lens')
            self._data = x
            self._starts = starts
            self._lens = lens

        assert(len(self._starts)==len(self._lens))
        self.shape = (len(self._starts),np.max(self._lens))
    
    def unpack(self):
        """
        unpack to list of arrays
        """
        return unpack_array(self._data,self._starts,self._lens)


    def applyfunc(self,func,*args,**kwargs):
        """
        apply function to all arrays

        Parameters
        ----------
        func: funciton
          signature of function is func(x,*args,**kwargs), where
          x is a subarray

        *args: function positional arguments to func

        **kwargs: keyword arguments to func

        Returns
        -------
        y: list
          func applied to each subarray
        """

        return [func(x,*args,**kwargs) for x in self.unpack()]

    def get_filled(self,**kwargs):
        """
        unpack to padded ndarray
        """
        return get_filled_array(self.unpack(),self.shape,**kwargs)
    
        
    def get_masked(self,**kwargs):
        """
        unpack to masked
        """
        return get_masked_array(self.unpack(),self.shape)


    def _get_single_array(self,i):
        """
        get single array

        Parameters
        ----------
        i: int
          index

        Returns
        -------
        y: ndarray
        """
        s = self._starts[i]
        c = self._lens[i]
        
        return self._data[s:s+c]
        
    def _get_single(self,i,j):
        """
        get single point

        Parameters
        ----------
        i,j: ints
         indices
         
        Returns
        -------
        y: value
         at (i,j)
        """

        # s = self._starts[i]
        # c = self._lens[i]

        # if j>=c:
        #     raise IndexError('shape of this segment is',(c,))

        # ii = s + j
        # return self._data[ii]

        return self._get_single_array(i)[j]
    
    def __getitem__(self,keys):
        if isinstance(keys,int):
            return self._get_single_array(keys)
        if len(keys)==2:
            if type(keys[-1]) is int:
                return self._get_single(*keys)
            elif type(keys[-1]) is slice:
                return self._get_single_array(keys[0])[keys[1]]
            else:
                raise IndexError('bad key')

        raise IndexError('bad key')

        
        

        
    
            
def pack_list(L):
    """
    pack a list of arrays to a single array

    Parameters
    ----------
    L: list [x1,x2,x3]
      list of arrays to pack

    Returns
    -------
    x: array
     packed 1d array

    starts: array
     start of each subarray

    lens: array
     length of each subarray

    """

    lens=np.array([len(x) for x in L])
    starts=np.empty(len(lens),dtype=int)
    starts[0]=0
    starts[1:]=np.cumsum(lens[:-1])
    
    x=np.concatenate(L)

    return x,starts,lens


def unpack_array(x,starts,lens):
    """
    unpack 1d array to list of subarrays

    Parameters
    ----------
    x: array
      1d array to unpack

    starts: array
      start of each subarray

    lens: array
      lenght of each subarray

    Returns
    -------
    L: list of subarrays
    """
    L=[]
    for (s,c) in izip(starts,lens):
        L.append(x[s:s+c])
    return L




#--------------------------------------------------
#tools to convert list of ragged arrays to single array padded with
#nan's


from itertools import izip_longest

def _find_shape(seq):
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [_find_shape(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,fillvalue=1))

def _fill_array(arr, seq, fill=np.nan):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = fill
    else:
        for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
            _fill_array(subarr, subseq, fill)

def get_filled_array(x,shape=None,dtype=float,fill=np.nan):
    """
    get a nan padded array
    """
    if shape is None:
        shape = _find_shape(x)
        
    y = np.empty(shape,dtype=dtype)

    _fill_array(y,x,fill)

    return y

def get_masked_array(x,shape=None,dtype=float):
    y = get_filled_array(x,shape)
    y = np.ma.array(y,mask=np.isnan(y)).astype(dtype)
    return y

# from itertools import izip_longest

# def _find_shape(seq):
#     try:
#         len_ = len(seq)
#     except TypeError:
#         return ()
#     shapes = [_find_shape(subseq) for subseq in seq]
#     return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,
#                                                                 fillvalue=1))

# def _fill_array(arr, seq, fill=np.nan):
#     if arr.ndim == 1:
#         try:
#             len_ = len(seq)
#         except TypeError:
#             len_ = 0
#         arr[:len_] = seq
#         arr[len_:] = fill
#     else:
#         for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
#             _fill_array(subarr, subseq, fill)

            
# def get_filled_array(x,dtype=float,fill=np.nan):
#     """
#     turn a list of lists to filled numpy.ndarray

#     *parameters*
#     x: list of lists to convert

#     dtype: type of output (default float)

#     fill: fill values (default np.nan)

#     *Returns*
#     y: filled ndarray
#     """
#     y = np.empty(_find_shape(x),dtype=dtype)
    
#     _fill_array(y,x,fill)
    
#     return y

# def get_masked_array(x,dtype=float):
#     """
#     turn a list of lists to masked array

#     *parameters*
#     x: list of lists to convert

#     dtype: type of output (default float)

#     fill: fill values (default np.nan)

#     *Returns*
#     y: masked array
#     """


#     y = get_filled_array(x)
#     y = np.ma.array(y,mask=np.isnan(y)).astype(dtype)
#     return y
    
    
    
