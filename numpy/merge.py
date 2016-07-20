"""
helper routins for numpy
"""
import numpy as _np
import pandas as _pd
from .converters import array_string_rep as _array_string_rep



def merge_arrays(x_list,fmt=None,tol=1e-7,return_index=True,x_format=None):
    """merges 1d arrays
    
    input
    =====
    x_list:  list of 1d arrays to merge
    fmt:     format for string representation.
             if None (default), then no string conversion.
             Otherwise, covert arrays to string representation for merge.
             [can fix nearly equal issues]

    tol:     arrays are considered equal if
             same length, and max(abs(x1-x0))<tol

    x_format (default None):
             format for returned x_new


    return (bool):  return index (if true) or mask (if false)
            
    
    return
    ======
    x_new     :  merged x_list

    if return_index is True:
        index_list:  list of indicies for merging
                     order same as order of x_list
                     x_new=x_list[i][index_list[i]]
    else:
        mask_list:  merging mask for each array in x_list
                    x_new=x_list[i][mask_list[i]]
    
    """
    
    if len(x_list)==1:
        return x_list[0],[slice(None,None,None)]
    
    for x in x_list:
        assert x.ndim==1
        
    #check if arrays arrays are equal
    len0=len(x_list[0])
    equal=True
    for x in x_list[1:]:
        if len(x)!=len0:

            equal=False
            break
            
        err=_np.max(_np.abs(x-x_list[0]))
        if err>tol:
            equal=False
            break
    
    if equal:
        x=x_list[0]
        index_list=[slice(None,None,None)]*len(x_list)
    else:
        #print 'new grid'
        #get index
  
        #create series:
        _s_list=[]
        for x in x_list:
            if fmt is not None:
                x=_array_string_rep(x,fmt)
                
            _s_list.append(_pd.Series(range(len(x)),index=x))
        
        _df=_pd.concat(_s_list,axis=1)#.dropna()
        _df=_df.dropna()
        x=_df.index.values
        index_list=[]
        for i in range(len(x_list)):
            index_list.append(_df[i].astype(int).values)
    

    #reformat x?
    if x_format is not None:
        x=x.astype(x_format)

    #mask or index
    if return_index:
        L=index_list
    else:
        #return mask
        L=[]
        for i in range(len(x_list)):
            msk=_np.zeros(x_list[i].shape,_np.bool)
            msk[index_list[i]]=True
            L.append(msk)
        
    

    return x,L
        
