#!/usr/bin/env python
"""
some utility functions
"""


import fnmatch as _fn
import os as _os
import re as _re

def simple_find_files(pattern,root_directory="."):
    """
    finds file(s) under root  directory with unix patter

    Parameters
    ==========
    pattern: string
          unix pattern for file match

    root_directory: string (default=".")
          directory to start search

    Returns
    ========
    L: list of file paths meeting criteria
    """
    
    L=[]
    sep=_os.path.sep
    for root, dir, files in _os.walk(root_directory):
        for item in _fn.filter(files,pattern):
               L.append(root+sep+item)
    return L


def clean_path(path_list):
    """
    clean up and double // in list of paths
    """

    L=[_re.sub(r'//',r'/',x) for x in path_list]
    
    return L

if __name__=='__main__':
    print "simple_find_files(pattern='*.py',root_directory='..//')"
    L=simple_find_files('*.py','..//')

    for l in L:
        print l

    print "clean up path_list"
    L=clean_path(L)
    for l in L:
        print l
