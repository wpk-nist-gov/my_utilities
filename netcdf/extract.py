"""
helper routines for dealing with netCDF files
"""





def get_dimension_definition(t):
    """
    get dimension definitions.


    input: nc object

    
    returns dictionary {dim_name:dim_size...}
    """
    d={}
    for k,l in t.dimensions.iteritems():
        d[k]=len(l)
    return d


def get_variable_definition(t,var_name_list=None,return_type=False):
    """
    get variable definitions in nc file.

    input
    =====
    t:  nc object
    
    var_name_list:  list of variable names to consider
                    if None, consider all variables.

    return_type:    return dic_type if True


    output=(dic_dim, dic_type)
    ======
    dic_dim:  dictionary of dimension names for variable.
    
    dic_type: type of varialbe for each name (if return_type==True)
    """
    
    if var_name_list is None:
        var_name_list=t.variables.keys()

    dic_dim={}
    for k in var_name_list:
        dic_dim[k] =t.variables[k].dimensions

    if not return_type:
        return dic_dim
    else:
        dic_type={}
        for k in var_name_list:
            dic_type[k]=t.variables[k].dtype

        return dic_dim, dic_type
    




def get_attributes(t):
    """
    get attributes

    input:  t=nc object or nc variable

    output: dictionary of {name:value}
    """
    
    d={}
    for k in t.ncattrs():
        d[k]=t.getncattr(k)
    return d
    


