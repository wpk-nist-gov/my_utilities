import numpy as np


def CosTransform(y,dx,dk=None, \
                 direction=0,  \
                 forward_factor=2.0,reverse_factor=1./np.pi, \
                 return_k=False ):
    """
    cosine transform for the array y(x) at evenly spaced x to yhat(k)
    at evenly spaced k. Assumes x[0]=0.  Returns yhat(k), where len(yhat)=len(y), k[0]=0,
    and k is evenly spaced.  Note that len(x)-1 must be even.
    
    Input:
    -----
    y
        1d array to be transformed

    dx  
        spacing between x.
    dk=None 
        spacing between k. Note that if not specified (or is None), then
        dk=pi/(dx*(len(x)-1))
    
    direction=0 [int]
        if direction is 0, than no forward, or reverse factor is
        included.  if direction>0, forward_factor is used (x->k). If
        direction<0, reverse_factor (k->x) is used.

    forward_factor=2.0
        for infinite region (typical use)

    reverse_factor=1/np.pi
        for infinite region (i.e., 1/(2*pi) is normal reverse factor,
        but double it for infinite region)
    
    return_k=False
        if true, output k array


    Output:
    -----
    yhat
        1d tranformed array
    k
        if return_k is true, k array for yhat

    
    """



    nmax=len(y)-1
    if nmax%2 !=0 :
        print "ERROR: nmax not even"
        return None

    if dk is None:
        dk=np.pi/(dx*nmax)


    xmax=float(nmax)*dx
    

    tau_array=np.arange(0,len(y),1)*1.0
    yhat=np.zeros(len(y))

    #loop over omega
    for nu in range(len(y)):
        omega=nu*dk
        theta=omega*dx
        
        #filon params
        sinth = np.sin ( theta )
        costh = np.cos ( theta )
        sinsq = sinth * sinth
        cossq = costh * costh
        thsq  = theta * theta
        thcub = thsq * theta

        if theta == 0.0 :
            alpha = 0.0
            beta  = 2.0 / 3.0
            gamma = 4.0 / 3.0

        else:

            alpha = ( 1.0 / thcub ) \
                    * ( thsq + theta * sinth * costh - 2.0 * sinsq )
            beta  = ( 2.0 / thcub ) \
                    * ( theta * ( 1.0 + cossq ) -2.0 * sinth * costh )
            gamma = ( 4.0 / thcub ) * ( sinth - theta * costh )



        #** DO THE SUM OVER THE EVEN ORDINATES **
        ce=np.sum(y[::2]*np.cos(theta*tau_array[::2]))
        
        #C       ** SUBTRACT HALF THE FIRST AND LAST TERMS **
        #                 COS(0)=1
        ce = ce - 0.5 * ( y[0] + y[-1] * np.cos ( omega * xmax ) )
        
        #C       ** DO THE SUM OVER THE ODD ORDINATES **
        co=np.sum(y[1::2]*np.cos(theta*tau_array[1::2]))
        
        #                                        =0
        #YHAT=DX*(ALPHA*(Y[-1]*SIN(OMEGA*XMAX)-Y0*SIN(OMEGA*T0))+BETA*CE+GAMMA*CO)
        
        yhat[nu] =  ( alpha * (y[-1] * np.sin ( omega * xmax ) ) \
                      + beta * ce + gamma * co ) * dx
       
    if direction>0:
        yhat=yhat*forward_factor
    elif direction<0:
        yhat=yhat*reverse_factor

    if return_k:
        k=np.arange(len(yhat))*dk
        return yhat,k
    else: 
        return yhat



def SinTransform(y,dx,dk=None, \
                 direction=0,  \
                 forward_factor=2.0,reverse_factor=1./np.pi, \
                 return_k=False ):
    """
    sine transform for the array y(x) at evenly spaced x to yhat(k)
    at evenly spaced k.  Assumes x[0]=0.  Returns yhat(k), where len(yhat)=len(y), k[0]=0,
    and k is evenly spaced.  Note that len(x)-1 must be even.
    
    Input:
    -----
    y
        1d array to be transformed

    dx  
        spacing between x.
    dk=None 
        spacing between k. Note that if not specified (or is None), then
        dk=pi/(dx*(len(x)-1))
    
    direction=0 [int]
        if direction is 0, than no forward, or reverse factor is
        included.  if direction>0, forward_factor is used (x->k). If
        direction<0, reverse_factor (k->x) is used.

    forward_factor=2.0
        for infinite region (typical use)

    reverse_factor=1/np.pi
        for infinite region (i.e., 1/(2*pi) is normal reverse factor,
        but double it for infinite region)
    
    return_k=False
        if true, output k array


    Output:
    -----
    yhat
        1d tranformed array
    k
        if return_k is true, k array for yhat

    
    """

    """

    """

    nmax=len(y)-1
    if nmax%2 !=0 :
        print "ERROR: nmax not even"
        return None
        

    if dk is None:
        dk=np.pi/(dx*nmax)


    xmax=float(nmax)*dx

    tau_array=np.arange(0,len(y),1)*1.0
    yhat=np.zeros(len(y))

    #loop over omega
    for nu in range(len(y)):
        omega=nu*dk
        theta=omega*dx
        
        #filon params
        sinth = np.sin ( theta )
        costh = np.cos ( theta )
        sinsq = sinth * sinth
        cossq = costh * costh
        thsq  = theta * theta
        thcub = thsq * theta

        if theta == 0.0 :
            alpha = 0.0
            beta  = 2.0 / 3.0
            gamma = 4.0 / 3.0

        else:

            alpha = ( 1.0 / thcub ) \
                    * ( thsq + theta * sinth * costh - 2.0 * sinsq )
            beta  = ( 2.0 / thcub ) \
                    * ( theta * ( 1.0 + cossq ) -2.0 * sinth * costh )
            gamma = ( 4.0 / thcub ) * ( sinth - theta * costh )



        #** DO THE SUM OVER THE EVEN ORDINATES **
        se=np.sum(y[::2]*np.sin(theta*tau_array[::2]))
        
        #C       ** SUBTRACT HALF THE FIRST AND LAST TERMS **
        #                Y0*sin(0)=0
        se = se - 0.5 * (            y[-1] * np.sin ( omega * xmax ) )
        
        #C       ** DO THE SUM OVER THE ODD ORDINATES **
        so=np.sum(y[1::2]*np.sin(theta*tau_array[1::2]))
        
        #                   COS(0)=0.0
        #YHAT=DX*(ALPHA*(Y0*COS(OMEGA*T0)-Y[-1]*COS(OMEGA*XMAX))+BETA*SE+GAMMA*SO)
        yhat[nu] =  ( alpha * (y[0] -y[-1] * np.cos ( omega * xmax )) \
                      + beta * se + gamma * so ) * dx
       
    if direction>0:
        yhat=yhat*forward_factor
    elif direction<0:
        yhat=yhat*reverse_factor

    if return_k:
        k=np.arange(len(yhat))*dk
        return yhat,k
    else: 
        return yhat





        

    
