import numpy as np

#TODO add in covariance?


class RunningStats(object):
    """
    object to keep running stats
    """
    
    def __init__(self,dtype=float):
        self._dtype = dtype
        self.Zero()
    
    def Zero(self):
        #spliced statistics
        #S[0] -> mean
        #S[1] -> var
        self._S = np.zeros(2,dtype=self._dtype)

        #variance in S
        #VS[0] -> variance in S[0] (mean)
        #VS[1] -> variance in S[1] (var)
        self._VS = [0.0,np.zeros(2,dtype=self._dtype)]

        #Weights
        #W[0] -> sum weights
        #W[1] -> sum weights**2
        self._W = np.zeros(2,dtype=self._dtype)
        

    def PushAveVar(self,w,ave,var):
        W0_last = self._W[0]
        self._W[0] += w
        self._W[1] += w*w

        f = np.empty(2,dtype=self._dtype)
        f[0] = w/self._W[0]
        f[1] = W0_last * f[0]
        
        delta = np.empty(2,dtype=self._dtype)
        delta[0] = ave - self._S[0]
        delta[1] = delta[0]**2

        #spliced stats
        other = np.array([0.0,var*w])
        self._S += other +delta*f


        #variance in spliced stats
        self._VS[0] += delta[1]*f[1]

        delta[0] = (var - self._VS[1][0])
        delta[1] = delta[0]*delta[0]

        self._VS[1] += delta*f
        


    def __add__(self,b):
        W0_last = self._W[0]
        
        combined = RunningStats(dtype=self._dtype)
        combined._W = self._W + b._W

        f = np.empty(2,dtype=self._dtype)
        f[0] = b._W[0]/combined._W[0]
        f[1] = W0_last * f[0]
        
        delta = np.empty(2,dtype=self._dtype)
        delta[0] = b._S[0] - self._S[0]
        delta[1] = delta[0]**2

        #spliced stats
        other = np.array([0.0,b._S[1]])
        combined._S = self._S + other + delta*f


        #var in splice
        combined._VS[0] = self._VS[0] + b._VS[0] + delta[1]*f[1]

        delta[0] = b._VS[1][0] - self._VS[1][0]
        delta[1] = delta[0]**2
        other = np.array([0.0,b._VS[1][1]])
        combined._VS[1] = self._VS[1] + other + delta*f
        

        return combined


    def __iadd(self,b):
        return self + b
        
        
    def PushArray(self,x):
        self.PushAveVar(float(len(x)),np.mean(x),np.var(x))
        
    def PushSingle(self,x):
        self.PushAveVar(1.,x,0.0)


    @property
    def Mean(self):
        return self._S[0]

    @property
    def Var(self):
        return self._S[1]/self._W[0]
    
    def Std(self,d=0.0):
        fac = self._W[0]/(self._W[0] - d)
        return np.sqrt(self.Var*fac)

    @property
    def VarInMean(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._VS[0] *fac

    @property
    def MeanOfVar(self):
        return self._VS[1][0]

    @property
    def VarInVar(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._VS[1][1] *fac





class RunningStatsVec(object):
    """
    object to keep running stats of vector object
    """
    
    def __init__(self,shape,dtype=float):
        self._dtype = dtype
        self._shape = shape
        self.Zero()
    
    def Zero(self):
        #spliced statistics
        #S[0,:] -> mean
        #S[1,:] -> var
        self._S = np.zeros((2,)+self._shape,dtype=self._dtype)

        #variance in S
        #VS[0,:] -> variance in S[0] (mean)
        #VS[1,:] -> variance in S[1] (var)
        self._VS = [np.zeros(self._shape,dtype=self._dtype),
                    np.zeros((2,)+self._shape,dtype=self._dtype)]

        #Weights
        #W[0] -> sum weights
        #W[1] -> sum weights**2
        self._W = np.zeros(2,dtype=self._dtype)
        

    def PushAveVar(self,w,ave,var):
        W0_last = self._W[0]
        self._W[0] += w
        self._W[1] += w*w


        f = w/self._W[0]
        delta = np.empty((2,)+self._shape,dtype=self._dtype)
        delta[0,...] = ave - self._S[0,...]
        delta[1,...] = delta[0,...]**2
        delta[0,...] *= f
        delta[1,...] *= f*W0_last

        #spliced stats
        self._S += delta
        self._S[1] += var*w
        
        # self._S[0,...] += delta[0,...]*f[0]
        # self._S[1,...] += var*w + delta[1,...]*f[1]
        #other = np.array([0.0,var*w])        
        #self._S += other +delta*f


        #variance in spliced stats
        self._VS[0][:] += delta[1,...]

        delta[0,...] = (var - self._VS[1][0,...])
        delta[1,...] = delta[0,...]**2
        delta[0,...] *= f
        delta[1,...] *= f*W0_last
        
        self._VS[1] += delta
        


    def __add__(self,b):
        W0_last = self._W[0]
        
        combined = RunningStatsVec(shape=self._shape,dtype=self._dtype)
        combined._W = self._W + b._W

        f = b._W[0]/combined._W[0]
        
        delta = np.empty((2,)+self._shape,dtype=self._dtype)
        delta[0,...] = b._S[0,...] - self._S[0,...]
        delta[1,...] = delta[0,...]**2
        delta[0,...] *= f
        delta[1,...] *= f*W0_last

        
        #spliced stats
        combined._S = self._S + delta
        combined._S[1,...] += b._S[1,...]
        
        #var in splice
        combined._VS[0] = self._VS[0] + b._VS[0] + delta[1,...]

        delta[0] = b._VS[1][0,...] - self._VS[1][0,...]
        delta[1] = delta[0,...]**2
        delta[0,...] *= f
        delta[1,...] *= f*W0_last

        combined._VS[1] = self._VS[1] + delta
        combined._VS[1][1,...] += b._VS[1][1,...]
        
        return combined


    def __iadd(self,b):
        return self + b
        
        
    def PushArray(self,x,axis=-1):
        self.PushAveVar(float(x.shape[axis]),np.mean(x,axis=axis),np.var(x,axis=axis))
        
    def PushSingle(self,x):
        self.PushAveVar(1.,x,0.0)


    @property
    def Mean(self):
        return self._S[0]

    @property
    def Var(self):
        return self._S[1]/self._W[0]
    
    def Std(self,d=0.0):
        fac = self._W[0]/(self._W[0] - d)
        return np.sqrt(self.Var*fac)

    @property
    def VarInMean(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._VS[0] *fac

    @property
    def MeanOfVar(self):
        return self._VS[1][0]

    @property
    def VarInVar(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._VS[1][1] *fac




class RunningStatsVecCov(object):
    """
    object to keep running stats of vector object
    """
    
    def __init__(self,l,dtype=float):
        self._dtype = dtype
        self._len = l
        self._shape = (self._len,)
        self.Zero()
    
    def Zero(self):
        #spliced statistics
        #S[0,:] -> mean
        #S[1,:] -> var
        self._S = np.zeros((2,)+self._shape,dtype=self._dtype)

        self._C = np.zeros((self._len,self._len),dtype=self._dtype)


        #variance in S
        #VS[0,:] -> variance in S[0] (mean)
        #VS[1,:] -> variance in S[1] (var)
        self._VS = [np.zeros(self._shape,dtype=self._dtype),
                    np.zeros((2,)+self._shape,dtype=self._dtype)]

        #Weights
        #W[0] -> sum weights
        #W[1] -> sum weights**2
        self._W = np.zeros(2,dtype=self._dtype)
        

    def PushAveVar(self,w,ave,var,cov):
        W0_last = self._W[0]
        self._W[0] += w
        self._W[1] += w*w


        f = w/self._W[0]
        delta = np.empty((2,)+self._shape,dtype=self._dtype)
        delta[0,:] = ave - self._S[0,:]
        delta[1,:] = delta[0,:]**2

        deltaM = np.multiply.outer(delta[0,:],delta[0,:])*f*W0_last
        
        delta[0,:] *= f
        delta[1,:] *= f*W0_last

        #spliced stats
        self._S += delta
        self._S[1] += var*w

        self._C += cov*w + deltaM

        
        
        # self._S[0,:] += delta[0,:]*f[0]
        # self._S[1,:] += var*w + delta[1,:]*f[1]
        #other = np.array([0.0,var*w])        
        #self._S += other +delta*f


        #variance in spliced stats
        self._VS[0][:] += delta[1,:]

        delta[0,:] = (var - self._VS[1][0,:])
        delta[1,:] = delta[0,:]**2
        delta[0,:] *= f
        delta[1,:] *= f*W0_last
        
        self._VS[1] += delta
        


    def __add__(self,b):
        W0_last = self._W[0]
        
        combined = RunningStatsVec(shape=self._shape,dtype=self._dtype)
        combined._W = self._W + b._W

        f = b._W[0]/combined._W[0]
        
        delta = np.empty((2,)+self._shape,dtype=self._dtype)
        delta[0,:] = b._S[0,:] - self._S[0,:]
        delta[1,:] = delta[0,:]**2

        deltaM = np.multiply.outer(delta[0,:],delta[0,:])*f*W0_last
        
        delta[0,:] *= f
        delta[1,:] *= f*W0_last

        
        #spliced stats
        combined._S = self._S + delta
        combined._S[1,:] += b._S[1,:]

        combined._C = self._C + b._C + deltaM
        
        #var in splice
        combined._VS[0] = self._VS[0] + b._VS[0] + delta[1,:]

        delta[0] = b._VS[1][0,:] - self._VS[1][0,:]
        delta[1] = delta[0,:]**2
        delta[0,:] *= f
        delta[1,:] *= f*W0_last

        combined._VS[1] = self._VS[1] + delta
        combined._VS[1][1,:] += b._VS[1][1,:]
        
        return combined


    def __iadd(self,b):
        return self + b
        
        
    def PushArray(self,x,axis=-1,rowvar=1,ddof=0):
        self.PushAveVar(float(x.shape[axis]),np.mean(x,axis=axis),np.var(x,axis=axis,ddof=ddof),np.cov(x,rowvar=rowvar,ddof=ddof))
        
    def PushSingle(self,x):
        self.PushAveVar(1.,x,0.0,0.0)


    @property
    def Mean(self):
        return self._S[0]

    @property
    def Var(self):
        return self._S[1]/self._W[0]
    
    def Std(self,d=0.0):
        fac = self._W[0]/(self._W[0] - d)
        return np.sqrt(self.Var*fac)

    @property
    def VarInMean(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._VS[0] *fac

    @property
    def MeanOfVar(self):
        return self._VS[1][0]

    @property
    def VarInVar(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._VS[1][1] *fac

        
