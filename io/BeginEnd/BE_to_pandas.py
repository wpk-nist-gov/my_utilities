import pandas as pd
from cStringIO import StringIO as sIO


def _get_tag(_l):
    ctag=_l.strip().split(':')[-1].strip()
    return ctag
    
def _get_names(_d):
    _n=[]
    for key in sorted(_d.keys()):
        _n.append(_d[key])

    return _n


def read_header(f,END_pattern="#BEGIN",sep='\s+',prop_col_id='#PROP_COL'):
    """returns array of column names based on PROP_COL constrcut

    f:           file object
    END_pattern: pattern to stop read 
                 None=>read to end of file
                 "#BEGIN" default
    sep:         value separator
    prop_col_id: string defining prop/col
                 "#PROP_COL default
    """

    prop_dic={}


    for line in f: 
    
        if END_pattern in line:
            break

        if prop_col_id in line:
            col=int(line.strip().split()[-1])-1
            prop_dic[col]=line.split()[-2]
            

    names=_get_names(prop_dic)
    
    return names
    




def read_BE_to_dic_pandas_char(f,BEGIN_tag=1,END_tag=None,names=None,sep='\s+',prop_col_id='#PROP_COL'):
    """returns dictionary of DataFrames for BEGIN/END file structure
    
    f:         file object
    BEGIN_tag: tag to start reading at (default 1)
    END_tag:   tag to stop reading at (default None, read all)
    names:     list of column names.  if names='get', get names from
               prop_col_id construct
    sep:       column separator (default=\s+)
    prop_col_id:  str id for prop_col contruct (default='#PROP_COL')
    """
    
    
    df_dic={}
    prop_dic={}
    go=False
    str_f=""
    #names
    if names is None:
        got_names=True #search for names
    if isinstance(names,str):
        if 'get' in names.lower():
            got_names=False
            names=None
    if isinstance(names,list):
        got_names=True


    for line in f:
        if (not got_names) and (prop_col_id in line):
            col=int(line.strip().split()[-1])-1
            prop_dic[col]=line.split()[-2]

        if '#BEGIN' in line:
            ctag=_get_tag(line)

            #names
            if not got_names:
                got_names=True
                if len(prop_dic)>0: #no names found
                    names=_get_names(prop_dic)

            if ctag==str(BEGIN_tag).strip(): 
                go=True


            str_f=""
            continue

        if '#END' in line:
            if not go: continue #add in to dic

            if names is None:
                df_dic[ctag]=pd.read_csv(sIO(str_f),sep=sep,header=None)
            else:
                df_dic[ctag]=pd.read_csv(sIO(str_f),sep=sep,names=names)


            if END_tag is not None:
                c_t=_get_tag(line)
                if c_t==str(END_tag).strip(): break

            continue

        if len(line.strip())==0 or "#" in line:
            continue

        str_f+=line
            
            
    return df_dic



def read_BE_to_dic_pandas(f,BEGIN_tag=1,END_tag=None,names=None,sep='\s+',prop_col_id='#PROP_COL'):
    """returns dictionary of DataFrames for BEGIN/END file structure
    
    f:         file object
    BEGIN_tag: tag to start reading at (default 1)
    END_tag:   tag to stop reading at (default None, read all)
    names:     list of column names.  if names='get', get names from
               prop_col_id construct
    sep:       column separator (default=\s+)
    prop_col_id:  str id for prop_col contruct (default='#PROP_COL')
    """

    
    df_dic={}
    prop_dic={}
    go=False
    #names
    if names is None:
        got_names=True #search for names
    if isinstance(names,str):
        if 'get' in names.lower():
            got_names=False
            names=None
    if isinstance(names,list):
        got_names=True


    for line in f:
        if (not got_names) and (prop_col_id in line):
            col=int(line.strip().split()[-1])-1
            prop_dic[col]=line.split()[-2]

        if '#BEGIN' in line:
            ctag=_get_tag(line)
            
            

            #names
            if not got_names:
                got_names=True
                if len(prop_dic)>0: #no names found
                    names=_get_names(prop_dic)

            if ctag==str(BEGIN_tag).strip():
                go=True


            s=sIO()
            continue

        if '#END' in line:
            if not go: continue #add in to dic
            
            s.seek(0)
            if names is None:
                df_dic[ctag]=pd.read_csv(s,sep=sep,header=None)
            else:
                df_dic[ctag]=pd.read_csv(s,sep=sep,names=names)

            s.close()

            if END_tag is not None:
                c_t=_get_tag(line)
                if c_t==str(END_tag).strip(): break

            continue

        if len(line.strip())==0 or "#" in line:
            continue

        #str_f+=line
        s.write(line)
            
            
    return df_dic
            
            


def read_BE_to_pandas_merge(f,BEGIN_tag=1,END_tag=None,names=None,sep='\s+',prop_col_id='#PROP_COL',include_tag=False):
    """returns DataFrames for BEGIN/END file structure
    
    f:         file object
    BEGIN_tag: tag to start reading at (default 1)
    END_tag:   tag to stop reading at (default None, read all)
    names:     list of column names.  if names='get', get names from
               prop_col_id construct
    sep:       column separator (default=\s+)
    prop_col_id:  str id for prop_col contruct (default='#PROP_COL')
    include_tag:       include 'tag' column or not (default False)
    """

    
    prop_dic={}
    go=False
    _df=pd.DataFrame()
    #names
    if names is None:
        got_names=True #search for names
    if isinstance(names,str):
        if 'get' in names.lower():
            got_names=False
            names=None
    if isinstance(names,list):
        got_names=True


    for line in f:
        if (not got_names) and (prop_col_id in line):
            col=int(line.strip().split()[-1])-1
            prop_dic[col]=line.split()[-2]

        if '#BEGIN' in line:
            ctag=_get_tag(line)
            
            

            #names
            if not got_names:
                got_names=True
                if len(prop_dic)>0: #no names found
                    names=_get_names(prop_dic)

            if ctag==str(BEGIN_tag).strip():
                go=True


            s=sIO()
            continue

        if '#END' in line:
            if not go: continue #add in to dic

            s.seek(0)
            if names is None:
                _df_tmp=pd.read_csv(s,sep=sep,header=None)
                #df_dic[ctag]=pd.read_csv(s,sep=sep,header=None)
            else:
                _df_tmp=pd.read_csv(s,sep=sep,names=names)
                #df_dic[ctag]=pd.read_csv(s,sep=sep,names=names)
            
            s.close()
            
            if include_tag:_df_tmp['tag']=ctag
                
            _df=_df.append(_df_tmp)

            if END_tag is not None:
                c_t=_get_tag(line)
                if c_t==str(END_tag).strip(): break

            continue

        if len(line.strip())==0 or "#" in line:
            continue

        #str_f+=line
        s.write(line)
            
            
    return _df
            
