"""
pandas utilities
"""

import numpy as np
import pandas as pd

# def SafeMerge(left,right,how,on,fmt=None,verbose=False,**kwargs):
#     """
#     do a "safe" merge using a string conversion
#     """

#     if fmt is None:
#         fmt='%s'

#     if type(on) is str:
#         on=[on]

#     if type(fmt) is str:
#         #apply single fmt to all columns
#         fmt=[fmt]*len(on)

#     assert(len(fmt)==len(on))

#     l=left.copy()
#     r=right.copy()

#     cols=on
#     scols=['_s_%s'%x for x in cols]

#     for sc,c,f in zip(scols,cols,fmt):
#         l[c]=l[c].apply(lambda x: f%x)
#         r[c]=r[c].apply(lambda x: f%x)


def SafeMerge(left, right, on, reMapdtypes=True, how='left', **kwargs):
    """
    do a "safe" merge using a string conversion

    convert on columns to str, then merge

    *Parameters*

    left,right,how,on: merge parameters

    **kwargs: arguments to pandas.merge

    reMapdtypes: bool (default True)
      if True: remap colums to original format
      if False: keep columns as strings
      if how=='right', use right dtypes, else use left dtypes
    """

    if reMapdtypes:
        if how == 'right':
            OrigFmt = [str(x) for x in right.dtypes[on]]
        else:
            OrigFmt = [str(x) for x in left.dtypes[on]]

    l = left.copy()
    r = right.copy()

    l[on] = l[on].astype(str)
    r[on] = r[on].astype(str)

    m = pd.merge(l, r, how=how, on=on, **kwargs)

    if reMapdtypes:
        for c, f in zip(on, OrigFmt):
            m[c] = m[c].astype(f)

    return m


def interpolate_DataFrame(df,
                          x,
                          interp_column='dens',
                          method='linear',
                          maskminmax=True,
                          sort_index=True,
                          retAll=False,
                          **kwargs):
    """
    interpolate a dataframe at x

    *Parameters*

    df: DataFrame
      DataFrame to interpolate (all columns other than interp_column
      are interpolated)

    x: array
      values to interpolate at

    interp_column: str
      column to interpolate over

    method: str
      method for interpolation (see pandas.DataFrame.interpolate
      options)

    retAll: bool
      if True, return all.
      else only at interpolated points "x"

    assumes df is indexed in same way as index_df
    """

    #get min/max values along interp_column
    if maskminmax:
        minx = df[interp_column].min()
        maxx = df[interp_column].max()
        xx = x[(minx <= x) & (x <= maxx)]
    else:
        xx = x

    ## old way:
    # #make empty DataFrame.  Index only
    # interp_df = pd.DataFrame({interp_column: xx}).set_index(interp_column)

    # #add new rows
    # t = pd.concat((df.set_index(interp_column), interp_df))
    ## end old way

    ## new way
    # consider doing something like below:
    # if drop_duplicates:
    #     t = df.drop_duplicates()
    # else:
        # t = df

    # set index
    t = df.set_index(interp_column)

    # new index from union of indicies
    idx_df = t.index
    idx_x = pd.Index(xx, name=idx_df.name
    )
    idx_union = idx_df.union(idx_x)

    # reindex
    t = t.reindex(idx_union)
    # end new way

    if sort_index:
        t = t.sort_index()

    #interpolate over index (linear interpolation)
    t = t.interpolate('index')  

    if not retAll:
        t = t.reindex(idx_x)

    t = t.reset_index()

    return t


def interpolate_DataFrame_group(df,
                                x,
                                interp_column,
                                group_columns,
                                maskminmax=True,
                                sort_index=True,
                                retAll=False,
                                **kwargs):
    """
    interpolate dataframe along single column with grouping

    Parameters
    ----------
    df : Dataframe

    x : array-like
        values to interplate at

    interp_column : str
        name of column to interpolate along

    group_columns : str on list-like
        column name(s) to group by

    method : string
        method for interplation (see pandas.DataFrame.interpolate)

    retAll : bool
        if True, return all,
        else, return at interpolated points
    """

    L = []
    group_columns = list(np.atleast_1d(group_columns))

    for v, g in df.groupby(group_columns):

        # setup values to assign back to frame

        vv = np.atleast_1d(np.array(v, dtype=np.object))
        d = dict(zip(group_columns, vv))


        L.append(
            interpolate_DataFrame(
                df=g.drop(group_columns, axis=1),
                x=x,
                interp_column=interp_column,
                maskminmax=maskminmax,
                sort_index=sort_index,
                retAll=retAll,**kwargs
            )
            .assign(**d)
        )

    return pd.concat(L)






from ..numpy import stats


def StatsAggDataFrame(df,
                      gcol,
                      ycol,
                      functions=None,
                      collapse='_',
                      droplist=['mean'],
                      localSize=False,
                      globalSize='size'):
    """
    Perform common groupby aggregation

    Parameters
    ----------

    df: DataFrame
        frame to aggregate

    gcol: list
          columns to groupby

    ycol: list
          columns to aggregate over.
          if ycol==None, use all columns less gcol

    collapse: bool or str
      if 'True': collapse Multiindex columns to regular index with '_'
      if 'False' or None: do nothing
      if str: collapse Mutiindex with character collapse

    droplist: list like
     if 'False' (or evaluates False): nothing
     otherwise, loop over droplist and drop 'collapse'+droplist[i]
     from column names. (default=['mean'])

    funcitons: list
               list of functions to aggregate.
               if 'None', use [numpy.mean, StandardError]

    localSize: bool
      if 'True', include size for each ycol. functions=function+[np.size]
      if 'False', do nothing
      
    globalSize: 
      if type(globalSize) is str, then add a column of name
      'globalSize' with size of each group


    Returns
    -------

    dfA: DataFrame 
      Aggregated DataFrame 

    """

    if not functions:
        functions = [np.mean, stats.StandardError]

    if localSize:
        functions = functions + [np.size]

    if ycol is None:
        ycol = df.columns.tolist()
        for x in gcol:
            ycol.remove(x)

            #do aggregation
    t = df[gcol + ycol].groupby(gcol).agg(functions)

    #global size?
    if globalSize and type(globalSize) is str:
        s = df[gcol].groupby(gcol).size()
        s.name = globalSize
        #add in
        t.loc[s.index, globalSize] = s

    #collapse
    if collapse:
        #collapse column names
        if type(collapse) is str:
            c = collapse
        else:
            c = '_'

        #join and strip trailing separators
        t.columns = [c.join(x).rstrip(c) for x in t.columns]

        #drop '_mean'?
        if droplist:
            for d in droplist:
                t.columns = [x.replace(c + d, '') for x in t.columns]

    return t



from scipy.stats import t as tdist


def StatsAggDataFrame_2(df,
                        gcol,
                        ycol=None,
                        functions=None,
                        collapse='_',
                        droplist=['mean'],
                        drop_std=True,
                        drop_size=True,
                        conf=0.95):
    """
    Perform common stats on frame

    Parameters
    ----------

    df: DataFrame
        frame to aggregate

    gcol: list
          columns to groupby

    ycol: list
          columns to aggregate over.
          if ycol==None, use all columns less gcol

    collapse: bool or str
      if 'True': collapse Multiindex columns to regular index with '_'
      if 'False' or None: do nothing
      if str: collapse Mutiindex with character collapse

    droplist: list like
     if 'False' (or evaluates False): nothing
     otherwise, loop over droplist and drop 'collapse'+droplist[i]
     from column names. (default=['mean'])

    drop_std : bool (default True)
        if True, drop standard deviations

    drop_size : bool (default True)
        if True, drop local size


    conf : float or None:
    if not None, than 

    globalSize: 
      if type(globalSize) is str, then add a column of name
      'globalSize' with size of each group


    Returns
    -------

    dfA: DataFrame 
      Aggregated DataFrame 

    """

    functions = ['mean']

    if not drop_std or conf:
        functions.append('std')

    if not drop_size or conf:
        functions.append('count')

    if ycol is None:
        ycol = list(df.columns.drop(gcol))

    # group, mean, and swap
    t = df[gcol + ycol].groupby(gcol).agg(functions).swaplevel(i=-1,j=0,axis=1).sortlevel(axis=1)

    if conf:
        # calculate Standard error frame:
        SE = t['std'] / np.sqrt(t['count']) * tdist.isf(0.5 * (1. - conf), t['count'] - 1)

        # add in SE label
        SE = pd.concat({'SE':SE},axis=1)

        # add into frame
        t = pd.concat((t,SE),axis=1)

    if drop_size:
        t = t.drop('count',axis=1)

    if drop_std:
        t = t.drop('std',axis=1)

    # swap names back to end
    t = t.swaplevel(i=0,j=-1,axis=1)

    #collapse
    if collapse:
        #collapse column names
        if type(collapse) is str:
            c = collapse
        else:
            c = '_'
        #join and strip trailing separators
        t.columns = [c.join(x).rstrip(c) for x in t.columns]

        #drop '_mean'?
        if droplist:
            for d in droplist:
                t.columns = [x.replace(c + d, '') for x in t.columns]

    return t



