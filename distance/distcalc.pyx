import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    double sqrt(double t)
    
@cython.wraparound(False)
@cython.boundscheck(False)
def self_pairwise_dv(np.ndarray[np.double_t, ndim=2] pos,np.ndarray[np.double_t,ndim=1] region):
    """
    finds the pairwise separation vectors
    
    Parameters
    ----------
    pos : ndarray
        shape = (N,D)
        
    region : ndarray
        region length in each dimension.  shape=(D,)
        
    Returns
    -------
    dv: ndarray
        shape = (N*(N-1)/2,D)
        dv[ij,:] = pos[j,:] - pos[i,:]
    """

    cdef int n,D
    cdef int i, j, k, size
    cdef np.ndarray[np.double_t, ndim=2] ans
    cdef double dx
    cdef np.ndarray[np.double_t, ndim=1] regionH
    
    n = pos.shape[0]
    D = pos.shape[1]

    regionH = 0.5*region
    
    size = n*(n-1)/2
    ans = np.empty((size,D), dtype=pos.dtype)
    
    k = -1
    for i in range(n-1):
        for j in range(i+1,n):
            k += 1
            for d in range(D):
                dx = pos[j,d] - pos[i,d]
                if dx >= regionH[d]:
                    dx = dx - region[d]
                elif dx < -regionH[d]:
                    dx = dx + region[d]
                
                ans[k,d] = dx

    return ans



@cython.wraparound(False)
@cython.boundscheck(False)
def self_pairwise_dv_dr(np.ndarray[np.double_t, ndim=2] pos,np.ndarray[np.double_t,ndim=1] region):
    """
    finds the pairwise separation vectors and distance squared array
    
    Parameters
    ----------
    pos : ndarray
        shape = (N,D)
        
    region : ndarray
        region length in each dimension.  shape=(D,)
        
    Returns
    -------
    dv: ndarray
        shape = (N*(N-1)/2,D)
        dv[ij,:] = pos[j,:] - pos[i,:]
        
    dr: ndarray
        shape = (N*(N-1)/2,)
        dr[ij,:] = |pos[j,:] - pos[i,:]|
    """
    

    cdef int n,D
    cdef int i, j, k, size
    cdef np.ndarray[np.double_t, ndim=2] dv
    cdef np.ndarray[np.double_t, ndim=1] dr
    
    cdef double dx, rr
    cdef np.ndarray[np.double_t, ndim=1] regionH
    
    n = pos.shape[0]
    D = pos.shape[1]

    regionH = 0.5*region
    
    size = n*(n-1)/2
    dv = np.empty((size,D), dtype=pos.dtype)
    dr = np.empty(size,dtype=pos.dtype)
    
    k = -1
    for i in range(n-1):
        for j in range(i+1,n):
            k += 1
            rr = 0
            for d in range(D):
                dx = pos[j,d] - pos[i,d]
                if dx >= regionH[d]:
                    dx = dx - region[d]
                elif dx < -regionH[d]:
                    dx = dx + region[d]
                rr = rr + dx*dx
                
                dv[k,d] = dx
            dr[k] = sqrt(rr)
            
    return dv,dr



@cython.wraparound(False)
@cython.boundscheck(False)
def self_pairwise_dv_Matrix(np.ndarray[np.double_t, ndim=2] pos,np.ndarray[np.double_t,ndim=1] region):
    """
    finds the pairwise separation matrix and distance squared array
    
    Parameters
    ----------
    pos : ndarray
        shape = (N,D)
        
    region : ndarray
        region length in each dimension.  shape=(D,)
        
    Returns
    -------
    dv: ndarray
        shape = (N,N,D)
        dv[i,j,:] = pos[j,:] - pos[i,:]
        
    """

    cdef int n,D
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=3] dv
    
    cdef double dx
    cdef np.ndarray[np.double_t, ndim=1] regionH
    
    n = pos.shape[0]
    D = pos.shape[1]

    regionH = 0.5*region
    
    dv = np.zeros((n,n,D), dtype=pos.dtype)
    
    for i in range(n-1):
        for j in range(i+1,n):

            for d in range(D):
                dx = pos[j,d] - pos[i,d]
                if dx >= regionH[d]:
                    dx = dx - region[d]
                elif dx < -regionH[d]:
                    dx = dx + region[d]

                dv[i,j,d] =  dx
                dv[j,i,d] = -dx
                
            

    return dv

@cython.wraparound(False)
@cython.boundscheck(False)
def self_pairwise_dv_dr_Matrix(np.ndarray[np.double_t, ndim=2] pos,np.ndarray[np.double_t,ndim=1] region):
    """
    finds the pairwise separation matrix and distance squared array
    
    Parameters
    ----------
    pos : ndarray
        shape = (N,D)
        
    region : ndarray
        region length in each dimension.  shape=(D,)
        
    Returns
    -------
    dv: ndarray
        shape = (N,N,D)
        dv[i,j,:] = pos[j,:] - pos[i,:]
        
    dr: ndarray
        shape = (N,N)
        dr[i,j,:] = |pos[j,:] - pos[i,:]|
    """

    cdef int n,D
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=3] dv
    cdef np.ndarray[np.double_t, ndim=2] dr
    
    cdef double dx,rr
    cdef np.ndarray[np.double_t, ndim=1] regionH
    
    n = pos.shape[0]
    D = pos.shape[1]

    regionH = 0.5*region
    
    dv = np.zeros((n,n,D), dtype=pos.dtype)
    dr = np.zeros((n,n),dtype=pos.dtype)
    
    for i in range(n-1):
        for j in range(i+1,n):
            rr = 0.0
            for d in range(D):
                dx = pos[j,d] - pos[i,d]
                if dx >= regionH[d]:
                    dx = dx - region[d]
                elif dx < -regionH[d]:
                    dx = dx + region[d]

                rr = rr + dx*dx
                dv[i,j,d] =  dx
                dv[j,i,d] = -dx
               
            rr = sqrt(rr)
            dr[i,j] = rr
            dr[j,i] = rr
            

    return dv,dr

@cython.wraparound(False)
@cython.boundscheck(False)
def pairwise_dv(np.ndarray[np.double_t, ndim=2] pos0,
                   np.ndarray[np.double_t, ndim=2] pos1,
                   np.ndarray[np.double_t,ndim=1] region):
    """
    finds the pairwise separation matrix and distance squared array
    
    Parameters
    ----------
    pos0 : ndarray
        shape = (N0,D)
        
    pos1 : ndarray
        shape = (N1,D)
        
    region : ndarray
        region length in each dimension.  shape=(D,)
        
    Returns
    -------
    dv: ndarray
        shape = (N0,N1,D)
        dv[i,j,:] = pos1[j,:] - pos0[i,:]
    """

    cdef int n0,D0,n1,D1
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=3] ans
    cdef double dx
    cdef np.ndarray[np.double_t, ndim=1] regionH
    
    
    n0 = pos0.shape[0]
    D0 = pos0.shape[1]
    
    n1 = pos1.shape[0]
    D1 = pos1.shape[1]
    
    #assert(D0==D1)

    regionH = 0.5*region
    
    ans = np.zeros((n0,n1,D0), dtype=pos0.dtype)
    
    for i in range(n0):
        for j in range(n1):
            for d in range(D0):
                dx = pos1[j,d] - pos0[i,d]
                if dx >= regionH[d]:
                    dx = dx - region[d]
                elif dx < -regionH[d]:
                    dx = dx + region[d]
                
                ans[i,j,d] = dx

    return ans


@cython.wraparound(False)
@cython.boundscheck(False)
def pairwise_dv_dr(np.ndarray[np.double_t, ndim=2] pos0,
                   np.ndarray[np.double_t, ndim=2] pos1,
                   np.ndarray[np.double_t,ndim=1] region):
    """
    finds the pairwise separation matrix and distance squared array
    
    Parameters
    ----------
    pos0 : ndarray
        shape = (N0,D)
        
    pos1 : ndarray
        shape = (N1,D)
        
    region : ndarray
        region length in each dimension.  shape=(D,)
        
    Returns
    -------
    dv: ndarray
        shape = (N0,N1,D)
        dv[i,j,:] = pos1[j,:] - pos0[i,:]
        
    dr: ndarray
        shape = (N0,N1)
        dr[i,j,:] = |pos1[j,:] - pos0[i,:]|
    """
    

    cdef int n0,D0,n1,D1
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=3] dv
    cdef np.ndarray[np.double_t, ndim=2] dr
    
    cdef double dx,rr
    cdef np.ndarray[np.double_t, ndim=1] regionH
    
    
    n0 = pos0.shape[0]
    D0 = pos0.shape[1]
    
    n1 = pos1.shape[0]
    D1 = pos1.shape[1]
    
    assert(D0==D1)

    regionH = 0.5*region
    
    dv = np.zeros((n0,n1,D0), dtype=pos0.dtype)
    dr = np.zeros((n0,n1),   dtype=pos0.dtype)
    
    for i in range(n0):
        for j in range(n1):
            rr = 0.0
            for d in range(D0):
                dx = pos1[j,d] - pos0[i,d]
                if dx >= regionH[d]:
                    dx = dx - region[d]
                elif dx < -regionH[d]:
                    dx = dx + region[d]
                rr = rr + dx*dx
                
                dv[i,j,d] = dx
            dr[i,j] = sqrt(rr)

    return dv,dr




# @cython.wraparound(False)
# @cython.boundscheck(False)
# def self_distancePBCM2(np.ndarray[np.double_t, ndim=2] pos,np.ndarray[np.double_t,ndim=1] region):

#     cdef int n,D
#     cdef int i, j
#     cdef np.ndarray[np.double_t, ndim=3] ans
#     cdef double dx
#     cdef np.ndarray[np.double_t, ndim=1] regionH
    
#     n = pos.shape[0]
#     D = pos.shape[1]

#     regionH = 0.5*region
    
#     ans = np.zeros((n,n,D), dtype=pos.dtype)
    
#     for i in range(n-1):
#         for j in range(i+1,n):
#             for d in range(D):
#                 dx = pos[j,d] - pos[i,d]
#                 if dx >= regionH[d]:
#                     dx = dx - region[d]
#                 elif dx < -regionH[d]:
#                     dx = dx + region[d]
                
#                 ans[i,j,d] = dx
#                 ans[j,i,d] = -dx

#     return ans


# @cython.wraparound(False)
# @cython.boundscheck(False)
# def self_distancePBCM2_dr(np.ndarray[np.double_t, ndim=2] pos,np.ndarray[np.double_t,ndim=1] region):

#     cdef int n,D
#     cdef int i, j
#     cdef np.ndarray[np.double_t, ndim=3] dv
#     cdef np.ndarray[np.double_t, ndim=2] drSq
    
#     cdef double dx,rr
#     cdef np.ndarray[np.double_t, ndim=1] regionH
    
#     n = pos.shape[0]
#     D = pos.shape[1]

#     regionH = 0.5*region
    
#     dv = np.zeros((n,n,D), dtype=pos.dtype)
#     drSq = np.zeros((n,n),dtype=pos.dtype)
    
#     for i in range(n-1):
#         for j in range(i+1,n):
#             rr = 0.0
#             for d in range(D):
#                 dx = pos[j,d] - pos[i,d]
#                 if dx >= regionH[d]:
#                     dx = dx - region[d]
#                 elif dx < -regionH[d]:
#                     dx = dx + region[d]

#                 rr = rr + dx*dx
#                 dv[i,j,d] =  dx
#                 dv[j,i,d] = -dx
                
#             drSq[i,j] = rr
#             drSq[j,i] = rr
            

#     return dv,drSq

# @cython.wraparound(False)
# @cython.boundscheck(False)
# def self_distanceM1(np.ndarray[np.double_t, ndim=2] pos):

#     cdef int n,D
#     cdef int i, j
#     cdef np.ndarray[np.double_t, ndim=3] ans

#     n = pos.shape[0]
#     D = pos.shape[1]

#     ans = np.empty((n,n,D), dtype=pos.dtype)
    
#     for i in range(n):
#         for j in range(n):
#             for d in range(D):
#                 ans[i,j,d] = pos[j,d] - pos[i,d]

#     return ans

# @cython.wraparound(False)
# @cython.boundscheck(False)
# def self_distanceM2(np.ndarray[np.double_t, ndim=2] pos):

#     cdef int n,D
#     cdef int i, j
#     cdef np.ndarray[np.double_t, ndim=3] ans
#     cdef double dx 
    
#     n = pos.shape[0]
#     D = pos.shape[1]

#     ans = np.zeros((n,n,D), dtype=pos.dtype)
    
#     for i in range(n-1):
#         for j in range(i+1,n):
#             for d in range(D):
                
#                 dx = pos[j,d] - pos[i,d]
#                 ans[i,j,d] = dx
#                 ans[j,i,d] = -dx

#     return ans
