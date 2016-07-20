"""
weighted stats
see http://en.wikipedia.org/wiki/Mean_square_weighted_deviation
"""

#create weighted average
from scipy.stats.distributions import  t as _ST
import numpy as np









#--------------------------------------------------
#uses 
#var=s^2 = \frac{\sum_{i=1}^N w_i}{{(\sum_{i=1}^N w_i})^2 -
#{\sum_{i=1}^N w_i^2} } \ . \ {\sum_{i=1}^N w_i (x_i -
#\overline{x}^{\,*})^2}
#
#i.e., sum(w*(x-mean)**2)

def weighted_average(w_in,y_in,conf=None,do_std=True):
    """computes weighted average of y with weight w
    over axis x
    
    input
    =====
    y_in: array or list of arrays to average.
          each element of the list is a numpy array
          each of these arrays has the structure (rec,val,....,x)
          where [val==0 is the mean] and [val==1 is the variance]
         

    w_in:    array of weights.
          shape is (nrec,x)
          
    conf:  confidance interval to use.
           (e.g., conf=0.95 for 95% conf. interval)
           if None (default), weighted standard deviation returned
    
    do_std: Bool
            Flag calculation of standard deviation/error.
            Default=True.
    

    output
    ======
    WA:  weighted average.  If y_in is a list, WA is a list.
         WA[i].shape=y_in[i].shape[1:]

    WSTD: weighted stdard deviation/error.  If none calculated, WSTD=None
    

    notes
    =====
    """
    assert type(w_in) is np.ndarray
    #assert type(y_list) is list
    assert w_in.ndim==2

    if type(y_in) is list:
        y_list=y_in
    else:
        y_list=[y_in]

    for y in y_list:
        assert y.shape[0]==w_in.shape[0]
        assert y.shape[-1]==w_in.shape[-1]
        assert type(y) is np.ndarray

    # #right type
    # w=w.astype(np.float)
        
    
    #normalize w
    w=w_in/np.sum(w_in,axis=0)
    
    WA_list=[]
    sig_WA_list=None
    
    if do_std and w.shape[0]>2:
        Sww=np.sum(w*w,axis=0)
        sig_WA_list=[]
        
    
    

    for y in y_list:
        #type
        #y=y.astype(np.float)
        
        #reshape w for broadcasting
        #not strictly necessary, but prevents any odd things
        new_shape=[w.shape[0]]+[1]*(y.ndim-2)+[w.shape[-1]]
        w.shape=new_shape
        #Sw.shape=new_shape[1:]
        #Sww.shape=new_shape[1:]
        
        WA=np.sum(w*y,axis=0)
        WA_list.append(WA)
        
        
        #weighted std dev
        if w.shape[0]>2 and do_std:
            Sww.shape=new_shape[1:]

            
            delta=y-WA
            
            sig_WA=np.sum(w*delta*delta,axis=0)
            
            sig_WA=sig_WA/(1.-Sww)

            #msk out bad values?
            msk=(sig_WA>0) & (np.isfinite(sig_WA))
            #zero bad points
            sig_WA[~msk]=0.
            sig_WA=np.sqrt(sig_WA)
            
            sig_WA_list.append(sig_WA)
        
    #print y.shape,WA.shape
    #do SE?
    if conf is not None and sig_WA_list is not None:
        sT=_ST.isf((1.-conf)*0.5,w.shape[0]-1)
        for i in range(len(sig_WA_list)):
            sig_WA_list[i]*=sT
            
            
    
    return WA_list,sig_WA_list


def spliced_ave_var(w_in,y_in,conf=None):
    """computes variance from pieces y with weight w over first axis
    
    input
    =====
    w_in:    array of weights.
          shape is (nrec,x)

    y_in: array or list of arrays of ave/variance
          each array of shape (nrec,val,...,x)
          val=0 => average
          val=1 => variance

          
    conf:  confidance interval to use.
           (e.g., conf=0.95 for 95% conf. interval)
           if None (default), weighted standard deviation returned
    
    do_std: Bool
            Flag calculation of standard deviation/error.
            Default=True.
    

    output
    ======
    Y:  weighted average/variance.  If y_in is a list, Y is a list.
         Y[i].shape=y_in[i].shape[1:]

    WSTD: weighted stdard deviation/error between blocks
    

    notes
    =====
    """
    assert type(w_in) is np.ndarray
    #assert type(y_list) is list
    assert w_in.ndim==2

    if type(y_in) is list:
        y_list=y_in
    else:
        y_list=[y_in]

    for y in y_list:
        assert y.ndim>2
        assert y.shape[0]==w_in.shape[0]
        assert y.shape[1]==2
        assert y.shape[-1]==w_in.shape[-1]
        assert type(y) is np.ndarray

    # #right type
    # w=w.astype(np.float)

    nrec=w_in.shape[0]
    ny=len(y_list)
    
    #normalize w
    #w=w_in/np.sum(w_in,axis=0)
    weight=w_in/np.sum(w_in,axis=0)
    

    W1=np.zeros(weight.shape[1:]) #sum weight
    W2=np.zeros(weight.shape[1:]) #sum weight**2
    
    M_V_1=[] #splice mean,variance
    M_V_2=[] #variance mean,variance

    V1=[] #mean of variance (not returned)
    
    w_shape=[] #reshaper for each y in y_list
    for y in y_list:
        M_V_1.append(np.zeros(y.shape[1:]))
        M_V_2.append(np.zeros(y.shape[1:]))

        V1.append(np.zeros(y.shape[2:]))
        #         y less rec,val,x
        new_shape=[1]*(y.ndim-3)+[y.shape[-1]]
        w_shape.append(new_shape)

        
    #accumulate
    for rec in range(nrec):
        
        w=weight[rec,:]
        
        W1_last=W1.copy() #note, have copy to make sure not pointing
        W1+=w
        W2+=w*w

        for iy in range(ny):
            s=w_shape[iy]
            x=y_list[iy][rec,0,...]
            v=y_list[iy][rec,1,...]

            f0=(w/W1).reshape(s)
            f1=(W1_last*w/W1).reshape(s)

            delta=(x-M_V_1[iy][0,...])
            delta_2=delta*delta

            #splice mean/var
            M_V_1[iy][0,...]+=delta*f0 #((w/W1).reshape(s))
            M_V_1[iy][1,...]+=v*w+delta_2*f1 #(W1_last*w/W1).reshape(s)

            #variance of mean
            M_V_2[iy][0,...]+=delta_2*f1 #(W1_last*w/W1).reshape(s)

            #variance of variance
            delta=(v-V1[iy])
            V1[iy]+=delta*f0 #(w/W1).reshape(s)
            M_V_2[iy][1,...]+=delta*delta*f1 #(W1_last*w/W1).reshape(s)
                



    #normalize spliced variance
    for iy in range(ny):
        M_V_1[iy][1,...]/=(W1.reshape(w_shape[iy]))

    #normalize variance of mean, variance of variance
    if nrec<2:
        fac=np.zeros(W1.shape)
    else:
        fac=W1/(W1**2-W2)
        
    for iy in range(ny):
        s=w_shape[iy]
        M_V_2[iy][0,...]*=fac.reshape(s)
        M_V_2[iy][1,...]*=fac.reshape(s)

        #mask out bads
        if nrec>1:
            msk=(M_V_2[iy]>0)&(np.isfinite(M_V_2[iy]))
            M_V_2[iy][~msk]=0.
            M_V_2[iy]=np.sqrt(M_V_2[iy])
        else:
            M_V_2[iy]=np.zeros(M_V_2[iy].shape)
            

    #confidence interval
    if conf is not None and nrec>1:
        sT=_ST.isf((1.-conf)*0.5,nrec-1)/np.sqrt(nrec)
        for iy in range(ny):
            M_V_2[iy]*=sT
    

    return M_V_1,M_V_2
    



class RunningStats(object):
    def __init__(self,dtype=float):
        self._dtype = dtype
        self.Zero()
    
    def Zero(self):
        #M[0] -> mean
        #M[1] -> var in mean
        self._M = np.zeros(2,dtype=self._dtype)

        #spliced variance
        self._V = 0.0
        #MV[0] -> mean of variance
        #MV[1] -> var of variance
        self._MV = np.zeros(2,dtype=self._dtype)

        #W[0] -> sum weights
        #W[1] -> sum weights**2
        self._W = np.zeros(2,dtype=self._dtype)
        

    def PushAveVar(self,w,ave,var):
        W0_last = self._W[0]
        self._W[0] += w
        self._W[1] += w*w

        delta = np.empty(2,dtype=self._dtype)
        delta[0] = ave - self._M[0]
        delta[1] = delta[0]**2
        # delta = (ave-self._M[0])
        # delta_2 = delta*delta


        f = np.empty(2,dtype=self._dtype)
        f[0] = w/self._W[0]
        f[1] = W0_last * f[0]

        #spliced mean
        self._M += delta*f

        #variance
        self._V += var*w + delta[1]*f[1]

        #var in M
        delta[0] = (var - self._MV[0])
        delta[1] = delta[0]**2
        self._MV += delta*f
        
        # #splice mean
        # self._M[0] += delta*f0
        # #var in mean
        # self._M[1] += delta_2*f1


        # #splice variance
        # self._V += var*w + delta_2*f1
        
        
        # #variance in M
        # delta = (var - self._MV[0])
        # delta_2 = delta*delta
        
        # self._MV[0] += delta*f0
        # self._MV[1] += delta_2*f1


    def __add__(self,b):
        W0_last = self._W[0]
        
        combined = RunningStats(dtype=self._dtype)
        combined._W = self._W + b._W

        f = np.empty(2,dtype=self._dtype)
        f[0] = b._W[0]/combined._W[0]
        f[1] = W0_last * f[0]
        
        delta = np.empty(2,dtype=self._dtype)

        delta[0] = b._M[0] - self._M[0]
        delta[1] = delta[0]**2
        
        other = np.array([0.0,b._M[1]])

        combined._M = self._M + other + delta*f
        # combined._M[0] = self._M[0] + delta[0]*f[0]
        # combined._M[1] = self._M[1] + b._M[1] + delta[1]*f[1]
        
        combined._V = self._V + b._V + delta[1]*f[1]


        delta[0] = b._MV[0] - self._MV[0]
        delta[1] = delta[0]**2
        other = np.array([0.0,b._MV[1]])
        combined._MV = self._MV + other + delta*f
        
        # combined._MV[0] = self._MV[0] + delta[0]*f[0]
        # combined._MV[1] = self._MV[1] + b._MV[1] + delta[1]*f[1]

        return combined


    def __iadd(self,b):

        return self + b
        
        
    def PushArray(self,x):
        self.PushAveVar(float(len(x)),np.mean(x),np.var(x))
        
    def PushSingle(self,x):
        self.PushAveVar(1.,x,0.0)

    def Mean(self):
        return self._M[0]

    
    def Var(self):
        return self._V/self._W[0]
    
    def Std(self,d=0.0):
        fac = self._W[0]/(self._W[0] - d)
        return np.sqrt(self.Var()*fac)

    def VarInMean(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._M[1] *fac
    
    def VarInVar(self):
        fac = self._W[0] / (self._W[0]**2 - self._W[1])
        return self._MV[1] *fac

    




##uses mean(y**2)-[mean(y)]**2
# def weighted_average(w_in,y_in,conf=None,do_std=True):
#     """computes weighted average of y with weight w
#     over axis x
    
#     input
#     =====
#     y_in: array or list of arrays to average.
#           each array of shape (nrec,...,x)

#     w_in:    array of weights.
#           shape is (nrec,x)
          
#     conf:  confidance interval to use.
#            (e.g., conf=0.95 for 95% conf. interval)
#            if None (default), weighted standard deviation returned
    
#     do_std: Bool
#             Flag calculation of standard deviation/error.
#             Default=True.
    

#     output
#     ======
#     WA:  weighted average.  If y_in is a list, WA is a list.
#          WA[i].shape=y_in[i].shape[1:]

#     WSTD: weighted stdard deviation/error.  If none calculated, WSTD=None
    

#     notes
#     =====
#     """
#     assert type(w_in) is _np.ndarray
#     #assert type(y_list) is list
#     assert w_in.ndim==2

#     if type(y_in) is list:
#         y_list=y_in
#     else:
#         y_list=[y_in]

#     for y in y_list:
#         assert y.shape[0]==w_in.shape[0]
#         assert y.shape[-1]==w_in.shape[-1]
#         assert type(y) is _np.ndarray

#     # #right type
#     # w=w.astype(np.float)
        
    
#     #normalize w
#     w=w_in/_np.sum(w_in,axis=0)
    
#     WA_list=[]
#     sig_WA_list=None
    
#     #Sw=_np.sum(w,axis=0) #=1.0 with norm
#     if do_std and w.shape[0]>2:
#         Sww=_np.sum(w*w,axis=0)
#         sig_WA_list=[]
        
    
    

#     for y in y_list:
#         #type
#         #y=y.astype(_np.float)
        
#         #reshape w for broadcasting
#         #not strictly necessary, but prevents any odd things
#         new_shape=[w.shape[0]]+[1]*(y.ndim-2)+[w.shape[-1]]
#         w.shape=new_shape
#         #Sw.shape=new_shape[1:]
#         Sww.shape=new_shape[1:]
        
#         Swy=_np.sum(w*y,axis=0)
        
#         WA=Swy #/Sw
#         WA_list.append(WA)
        
        
#         #weighted std dev
#         if w.shape[0]>2 and do_std:
#             Sww=_np.sum(w*w,axis=0)
#             Swyy=_np.sum(y*y*w,axis=0)
        
#             #sig_WA=_np.sqrt((Swyy*Sw-Swy*Swy)/(Sw*Sw-Sww))
#             sig_WA=(Swyy-Swy*Swy)/(1.0-Sww)

#             #msk out bad values?
#             msk=(sig_WA>0) & (_np.isfinite(sig_WA))
#             #zero bad points
#             sig_WA[~msk]=0.
#             sig_WA=_np.sqrt(sig_WA)
            
#             sig_WA_list.append(sig_WA)
        
#     #do SE?
#     if conf is not None and sig_WA_list is not None:
#         sT=_ST.isf((1.-conf)*0.5,w.shape[0]-1)
#         for i in range(len(sig_WA_list)):
#             sig_WA_list[i]*=sT
            
            
    
#     return WA_list,sig_WA_list

