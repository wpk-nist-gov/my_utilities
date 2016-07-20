#from .. import utilities as my_utils



from ..numpy import stats as st

def f():
    print "hello"

    print st.StandardError([1.,2.,3.])


import os

def g():
    print __file__
    print os.path.abspath(__file__)
    print os.path.abspath('.')
