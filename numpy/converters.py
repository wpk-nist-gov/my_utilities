import numpy as _np

def array_string_rep(x,fmt='%.3f'):
    """
    return string representation of numpy array

    input
    =====
    x: numpy array
    fmt:  format for string representation (default='%.3f')

    output
    ======
    y: numpy array of strings
    """
    
    return _np.char.mod(fmt, x)
