"""
some common utilities
"""


import os as _os
from collections import OrderedDict as _od
import ast as _ast

def get_file_description_dic(fname,ext,name_key='name',id_key='rn',\
                             sep='_', lower_keys=False, lower_values=False,
                             path=True):
    """
    convert fname to dictionary

    assume fname is in form:
       name_id_key0_val0_key1_val1_...._.ext

    Parameters
    ==========
    fname: string
    ext: string (will be stripped)
    
    name_key: key form name (=None=> no return)
    id_key:  key for id
    sep:  separator between keys/values (default="_")

    lower_keys: bool (default False)
          downcase keys

    lower_values: vool (default False)
          downcase values
    
    path : bool (default True)
    if True, include path

    Returns
    =======
    dic of {key:val...}
    """
    
    y=_os.path.basename(fname).strip(ext).split('_')

    d=_od()

    if name_key is not None:
        d[name_key]=y[0]


    d[id_key]=y[1]
    
    #add in reset
    k=y[2::2]
    if lower_keys:
        k=[x.lower() for x in k]

    v=y[3::2]
    if lower_values:
        v=[x.lower() for x in v]
        
    
    d.update(_od( zip(k,v) ))

    #convert to int or float
    for k in d:

        d[k]=try_string_conversion(d[k])
        # try:
        #     d[k]=_ast.literal_eval(d[k])
        # except:
        #     pass

    if path:
        d['path'] = fname
            
    return d


def try_string_conversion(x):
    """
    single string conversion to int/float

    if fails, returns string
    """
    try:
        y=_ast.literal_eval(x)
    except:
        y=x

    return y
    
    



if __name__=='__main__':
    
    fname='HS_1-A_Temp_1.0_Dens_0.95_n_1000_Some_200.00abc_.a.b.c'
    ext='.a.b.c'
    print 'fname',fname
    print 'ext',ext


    d=get_file_description_dic(fname,ext,lower_keys=False)

    print d

    for (k,v) in d.iteritems():
        print k,v,type(v)
    

    print "lower_keys=True"
    d=get_file_description_dic(fname,ext,lower_keys=True)

    print d

    for (k,v) in d.iteritems():
        print k,v,type(v)
