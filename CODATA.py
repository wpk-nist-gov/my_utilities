from scipy.constants import physical_constants as _codata


class constant(float):
    def __new__(cls,float_string,*args,**kwargs):
        return float.__new__(cls,float_string)
    
    def __init__(self,val,units=None,precision=None):
        self._val=str(val)
        self._units = units
        self._precision = precision
        
    @property
    def units(self):
        return self._units
    @property
    def precision(self):
        if self._precision is None:
            return 0.0
        else:
            return self._precision

    def __iter__(self):
        return iter((float(self._val),self.units,self.precision))

    def to_tuple(self):
        return tuple(self)

    def __repr__(self):
        return "(%s,'%s',%s)"%(self._val,self.units,self.precision)

    def __str__(self):
        return self.__repr__()
    
    def _repr_html_(self):
        s = self._val
        if self.precision is not None:
            s += ' +/- %s'%self.precision
        s += ' [%s]'%self.units
        
        return s


constants = {k:constant(*v) for k,v in _codata.iteritems()}
    
            
